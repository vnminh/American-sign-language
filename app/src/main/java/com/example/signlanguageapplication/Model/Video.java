package com.example.signlanguageapplication.Model;

import androidx.room.Entity;
import androidx.room.PrimaryKey;

@Entity(tableName = "videos")
public class Video {
    @PrimaryKey(autoGenerate = true)
    private int id;
    private int signId;
    private String videoBase64; // Sử dụng video_base64 từ Firestore
    private String filename;    // Sử dụng filename từ Firestore

    public Video(int signId, String videoBase64, String filename) {
        this.signId = signId;
        this.videoBase64 = videoBase64;
        this.filename = filename;
    }

    public int getId() { return id; }
    public void setId(int id) { this.id = id; }
    public int getSignId() { return signId; }
    public void setSignId(int signId) { this.signId = signId; }
    public String getVideoBase64() { return videoBase64; }
    public void setVideoBase64(String videoBase64) { this.videoBase64 = videoBase64; }
    public String getFilename() { return filename; }
    public void setFilename(String filename) { this.filename = filename; }
}