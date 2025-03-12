package com.example.signlanguageapplication.ViewModel;

import android.util.Log;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;
import androidx.lifecycle.ViewModel;

import com.example.signlanguageapplication.Model.SignRecognitionResult;

import java.io.IOException;
import java.io.InputStream;
import java.net.Socket;
import java.util.ArrayList;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class SignRecognitionViewModel extends ViewModel {

    private final MutableLiveData<ArrayList<SignRecognitionResult>> signList = new MutableLiveData<ArrayList<SignRecognitionResult>>(new ArrayList<>());
    private final ExecutorService executorService = Executors.newSingleThreadExecutor();
    private Socket socket;
    private InputStream inputStream = null;
    private final String SERVER_ADDRESS = "192.168.1.13";
    private final int SERVER_PORT = 6969;

    public LiveData<ArrayList<SignRecognitionResult>> getSignList() {
        return this.signList;
    }

    public void connectToServer() {
        executorService.execute(() -> {
            try {
                socket = new Socket(SERVER_ADDRESS, SERVER_PORT);
                inputStream = socket.getInputStream();
                receiveData();
            } catch (IOException e) {
                e.printStackTrace();
            }
        });
    }

    public void receiveData() {
        executorService.execute(() -> {
            try {
                int byteRead;
                byte[] buffer = new byte[1024];

                while((byteRead = inputStream.read(buffer)) != -1) {
                    String receiveData = new String(buffer, 0, byteRead);
                    Log.d("Client nhận: ",receiveData);

                    String[] parts = receiveData.split("\\|");

                    if(parts.length == 2) {
                        String signName = parts[0].trim();
                        String timestamp = parts[1].trim();
                        Log.d("SignName", signName);
                        Log.d("Timestamp", timestamp);
                        SignRecognitionResult result = new SignRecognitionResult(signName, timestamp);
                        ArrayList<SignRecognitionResult> currentSignList = new ArrayList<>(this.signList.getValue());

                        currentSignList.add(0, result);
                        this.signList.postValue(currentSignList);    // Đẩy dữ liệu từ luồng nên ExecutorService lên UI
                        Log.d("Client cập nhật", "Đã cập nhật");
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        });
    }

    @Override
    protected void onCleared() {
        super.onCleared();
        executorService.shutdown();
        try {
            if (socket != null) socket.close();
            if (inputStream != null) inputStream.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
    }
}
