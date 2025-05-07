package com.example.signlanguageapplication.ViewModel;

import android.app.Application;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.signlanguageapplication.DAO.SignRecognitionDao;
import com.example.signlanguageapplication.Database.SignRecognitionDatabase;
import com.example.signlanguageapplication.Model.SignRecognitionResult;

import java.io.IOException;
import java.io.InputStream;
import java.net.Socket;
import java.net.SocketException;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.Locale;
import java.util.Queue;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SignRecognitionViewModel extends AndroidViewModel {
    private final MutableLiveData<ArrayList<SignRecognitionResult>> signList = new MutableLiveData<>(new ArrayList<>());
    private final ExecutorService executorService = Executors.newSingleThreadExecutor();
    private final ExecutorService processingService = Executors.newSingleThreadExecutor();
    private Socket socket;
    private InputStream inputStream = null;
    private final String SERVER_ADDRESS = "192.168.1.18";
    private final int SERVER_PORT = 6969;
    private boolean isRunning = true;
    private final Queue<String> messageQueue = new LinkedList<>();
    
    private SignRecognitionDao recognitionDao;
    
    private TextToSpeech textToSpeech = null;

    private boolean isAutoSpeakEnabled = false; // Default to ON

    public SignRecognitionViewModel(Application application) {
        super(application);
        this.recognitionDao = SignRecognitionDatabase.getInstance(application).signRecognitionDao();
        textToSpeech = new TextToSpeech(application.getApplicationContext(), status -> {
            if (status == TextToSpeech.SUCCESS) {
                textToSpeech.setLanguage(Locale.US);
            }
        });
    }

    public LiveData<ArrayList<SignRecognitionResult>> getSignList() {
        return this.signList;
    }

    public void setSignList(ArrayList<SignRecognitionResult> currentSignList) {
        this.signList.postValue(currentSignList);
    }

    public void setAutoSpeakEnabled(boolean enabled) {
        isAutoSpeakEnabled = enabled;
    }

    public void connectToServer() {
        executorService.execute(() -> {
            while (isRunning) {
                try {
                    Log.d("Client", "Trying to connect to server...");
                    socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
                    inputStream = socket.getInputStream();
                    Log.d("Client", "Connected to server!");
                    receiveData();
                    processQueue();
                    break;
                } catch (IOException e) {
                    Log.e("Client", "Connection failed, retrying in 5 seconds...", e);
                    try {
                        Thread.sleep(5000);
                    } catch (InterruptedException ignored) {}
                }
            }
        });
    }

    public void receiveData() {
        executorService.execute(() -> {
            try {
                int byteRead;
                byte[] buffer = new byte[1024];

                while (isRunning && inputStream != null && (byteRead = inputStream.read(buffer)) != -1) {
                    String receiveData = new String(buffer, 0, byteRead);
                    Log.d("Client nhận: ", receiveData);

                    synchronized (messageQueue) {
                        messageQueue.add(receiveData);
                        messageQueue.notify();
                    }
                }
            } catch (SocketException e) {
                Log.e("Client", "Server disconnected. Attempting to reconnect...", e);
                closeSocket();
                connectToServer();
            } catch (IOException e) {
                Log.e("Client", "IO error while reading data", e);
            }
        });
    }

    private void processQueue() {
        processingService.execute(() -> {
            String currentString = "";
            boolean isNewLineAdded = false;

            while (isRunning) {
                String receiveData;
                synchronized (messageQueue) {
                    while (messageQueue.isEmpty() && isRunning) {
                        try {
                            messageQueue.wait();
                        } catch (InterruptedException e) {
                            Log.e("Client", "Queue processing interrupted", e);
                        }
                    }
                    if (!isRunning) break;
                    receiveData = messageQueue.poll();
                }

                if (receiveData != null) {
                    String[] parts = receiveData.split("\\|");
                    if (parts.length == 2) {
                        String signName = parts[0].trim();
                        String timestamp = parts[1].trim();
                        Log.d("SignName", signName);
                        Log.d("Timestamp", timestamp);

                        ArrayList<SignRecognitionResult> currentSignList = new ArrayList<>(this.signList.getValue());

                        if (signName.equals("end.")) {
                            if (!currentString.isEmpty()) {
                                if (!currentSignList.isEmpty() && isNewLineAdded) {
                                    currentSignList.set(0, new SignRecognitionResult(currentString.trim(), timestamp));
                                } else {
                                    currentSignList.add(0, new SignRecognitionResult(currentString.trim(), timestamp));
                                }

                                SignRecognitionResult result = new SignRecognitionResult(currentString.trim(), timestamp);
                                AsyncTask.execute(() -> {
                                    recognitionDao.insert(result);
                                    Log.d("DEBUG", "Insert data: " + result.toString());
                                });
                                if (isAutoSpeakEnabled) {
                                    textToSpeech.speak(currentString, TextToSpeech.QUEUE_FLUSH, null, null);
                                }
                                currentString = "";
                                isNewLineAdded = false;
                                this.signList.postValue(currentSignList);
                                Log.d("Client cập nhật", "Đã lưu câu hoàn chỉnh vào DB");
                            }
                        } else if (!isNewLineAdded) {
                            currentString += signName + " ";
                            currentSignList.add(0, new SignRecognitionResult(currentString, timestamp));
                            isNewLineAdded = true;
                            this.signList.postValue(currentSignList);
                            Log.d("Client cập nhật", "Đã thêm dòng mới: " + currentString);
                        } else {
                            currentString += signName + " ";
                            currentSignList.set(0, new SignRecognitionResult(currentString, timestamp));
                            this.signList.postValue(currentSignList);
                            Log.d("Client cập nhật", "Đã cập nhật chuỗi: " + currentString);
                        }
                    }
                }
            }
        });
    }

    private void closeSocket() {
        try {
            if (inputStream != null) {
                inputStream.close();
                inputStream = null;
            }
            if (socket != null) {
                socket.close();
                socket = null;
            }
        } catch (IOException e) {
            Log.e("Client", "Error closing socket", e);
        }
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        isRunning = false;
        executorService.shutdown();
        processingService.shutdown();
        if (textToSpeech != null) {
            textToSpeech.stop();
            textToSpeech.shutdown();
        }
        closeSocket();
    }
}