package com.example.signlanguageapplication.ViewModel;

import android.content.Context;
import android.os.AsyncTask;
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
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SignRecognitionViewModel extends ViewModel {
    private final MutableLiveData<ArrayList<SignRecognitionResult>> signList = new MutableLiveData<>(new ArrayList<>());
    private final ExecutorService executorService = Executors.newSingleThreadExecutor();
    private Socket socket;
    private InputStream inputStream = null;
    private final String SERVER_ADDRESS = "172.20.10.3";
    private final int SERVER_PORT = 6969;
    private boolean isRunning = true;  // Flag to keep track of server connection

    private SignRecognitionDao recognitionDao;

    public SignRecognitionViewModel(Context context) {
        this.recognitionDao = SignRecognitionDatabase.getInstance(context).signRecognitionDao();
    }
    public LiveData<ArrayList<SignRecognitionResult>> getSignList() {
        return this.signList;
    }

    public void setSignList(ArrayList<SignRecognitionResult> currentSignList) {
        this.signList.postValue(currentSignList);
    }

    public void connectToServer() {
        executorService.execute(() -> {
            while (isRunning) { // Keep trying to reconnect until ViewModel is cleared
                try {
                    Log.d("Client", "Trying to connect to server...");
                    socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
                    inputStream = socket.getInputStream();
                    Log.d("Client", "Connected to server!");
                    receiveData();
                    break; // Exit loop when connected successfully
                } catch (IOException e) {
                    Log.e("Client", "Connection failed, retrying in 5 seconds...", e);
                    try {
                        Thread.sleep(5000); // Wait before retrying
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

                    String[] parts = receiveData.split("\\|");

                    if (parts.length == 2) {
                        String signName = parts[0].trim();
                        String timestamp = parts[1].trim();
                        Log.d("SignName", signName);
                        Log.d("Timestamp", timestamp);
                        SignRecognitionResult result = new SignRecognitionResult(signName, timestamp);
                        ArrayList<SignRecognitionResult> currentSignList = new ArrayList<>(this.signList.getValue());

                        currentSignList.add(0, result);
                        this.signList.postValue(currentSignList);
                        AsyncTask.execute(new Runnable() {
                            @Override
                            public void run() {
//                                SignRecognitionViewModel.this.recognitionDao.insert(result);
                                recognitionDao.insert(result);
                                Log.d("DEBUG", "Insert data: " + result.toString());
                            }
                        });
                        Log.d("Client cập nhật", "Đã cập nhật");
                    }
                }
            } catch (SocketException e) {
                Log.e("Client", "Server disconnected. Attempting to reconnect...", e);
                closeSocket(); // Ensure socket is closed properly before reconnecting
                connectToServer();
            } catch (IOException e) {
                Log.e("Client", "IO error while reading data", e);
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
        isRunning = false; // Stop reconnect attempts
        executorService.shutdown();
        closeSocket();
    }
}
