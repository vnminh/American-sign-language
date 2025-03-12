package com.example.signlanguageapplication.Model;

public class SignRecognitionResult {
    private String signName;
    private String timestamp;

    public SignRecognitionResult(String signName, String timestamp) {
        this.signName = signName;
        this.timestamp = timestamp;
    }

    public String getSignName() {
        return signName;
    }

    public void setSignName(String signName) {
        this.signName = signName;
    }

    public String getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(String timestamp) {
        this.timestamp = timestamp;
    }
}