package com.example.signlanguageapplication.Model;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "sign_results")
public class SignRecognitionResult {
    @PrimaryKey(autoGenerate = true)
    private int id;
    private String signName;
    private String timestamp;

    public SignRecognitionResult(String signName, String timestamp) {
        this.signName = signName;
        this.timestamp = timestamp;
    }

    public int getId() { return id; }
    public void setId(int id) { this.id = id; }

    public String getSignName() { return signName; }
    public void setSignName(String signName) { this.signName = signName; }

    public String getTimestamp() { return timestamp; }
    public void setTimestamp(String timestamp) { this.timestamp = timestamp; }

    @Override
    public String toString() {
        return "SignRecognitionResult{" +
                "timestamp='" + timestamp + '\'' +
                ", signName='" + signName + '\'' +
                '}';
    }
}
