package com.example.signlanguageapplication.ViewModel;

import android.content.Context;
import android.os.AsyncTask;
import android.speech.tts.TextToSpeech;
import android.util.Log;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

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

public class SignRecognitionViewModel extends ViewModel {
    private final MutableLiveData<ArrayList<SignRecognitionResult>> signList = new MutableLiveData<>(new ArrayList<>());
    private final ExecutorService executorService = Executors.newSingleThreadExecutor();
    private final ExecutorService processingService = Executors.newSingleThreadExecutor(); // Luồng xử lý Queue
    private Socket socket;
    private InputStream inputStream = null;
    private final String SERVER_ADDRESS = "192.168.222.1";
    private final int SERVER_PORT = 6969;
    private boolean isRunning = true;
    private final Queue<String> messageQueue = new LinkedList<>(); // Queue để lưu trữ dữ liệu nhận được

    private SignRecognitionDao recognitionDao;

    private TextToSpeech textToSpeech = null;

    private boolean isAutoSpeakEnabled = false; // default to ON

    public void setAutoSpeakEnabled(boolean enabled) {
        isAutoSpeakEnabled = enabled;
    }

    private boolean isAutoSpeakEnabled() {
        return isAutoSpeakEnabled;
    }

    public SignRecognitionViewModel(Context context) {
        this.recognitionDao = SignRecognitionDatabase.getInstance(context).signRecognitionDao();
        textToSpeech = new TextToSpeech(context.getApplicationContext(), status -> {
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

    public void connectToServer() {
        executorService.execute(() -> {
            while (isRunning) {
                try {
                    Log.d("Client", "Trying to connect to server...");
                    socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
                    inputStream = socket.getInputStream();
                    Log.d("Client", "Connected to server!");
                    receiveData();
                    processQueue(); // Bắt đầu xử lý Queue
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
                        messageQueue.add(receiveData); // Thêm dữ liệu vào Queue
                        messageQueue.notify(); // Thông báo cho luồng xử lý
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
            String currentString = ""; // Buffer để lưu chuỗi hiện tại
            boolean isNewLineAdded = false; // Cờ để kiểm tra xem dòng mới đã được thêm chưa

            while (isRunning) {
                String receiveData;
                synchronized (messageQueue) {
                    while (messageQueue.isEmpty() && isRunning) {
                        try {
                            messageQueue.wait(); // Chờ nếu Queue rỗng
                        } catch (InterruptedException e) {
                            Log.e("Client", "Queue processing interrupted", e);
                        }
                    }
                    if (!isRunning) break;
                    receiveData = messageQueue.poll(); // Lấy dữ liệu từ Queue
                }

                if (receiveData != null) {
                    String[] parts = receiveData.split("\\|");
                    if (parts.length == 2) {
                        String signName = parts[0].trim();
                        String timestamp = parts[1].trim();
                        Log.d("SignName", signName);
                        Log.d("Timestamp", timestamp);

                        ArrayList<SignRecognitionResult> currentSignList = new ArrayList<>(this.signList.getValue());
                        Log.d("Current String", currentString);
                        Log.d("Cur list size", currentSignList.size() + "");

                        if (signName.equals("end.")) { // Ký hiệu kết thúc câu
                            Log.d("Điều kiện", "1");
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
                                if(isAutoSpeakEnabled) {
                                    textToSpeech.speak(currentString, TextToSpeech.QUEUE_FLUSH, null, null);
                                }
                                currentString = "";
                                isNewLineAdded = false;
                                this.signList.postValue(currentSignList);
                                Log.d("Client cập nhật", "Đã lưu câu hoàn chỉnh vào DB");
                            }       
                        } else if (!isNewLineAdded) {
                            currentString += signName + " ";
                            Log.d("Điều kiện", "2");
                            Log.d("Add first", "oke");
                            currentSignList.add(0, new SignRecognitionResult(currentString, timestamp));
                            Log.d("After Add first", "oke");
                            isNewLineAdded = true;
                            this.signList.postValue(currentSignList);
                            Log.d("Client cập nhật", "Đã thêm dòng mới: " + currentString);
                        } else if (isNewLineAdded) {
                            currentString += signName + " ";
                            Log.d("Điều kiện", "3");
                            currentSignList.set(0, new SignRecognitionResult(currentString, timestamp));
                            this.signList.postValue(currentSignList);
                            Log.d("Client cập nhật", "Đã cập nhật chuỗi: " + currentString);
                        }
                        Log.d("Đi hết đk", "ok");
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
        closeSocket();
    }
}