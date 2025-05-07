package com.example.signlanguageapplication.Service;

import android.content.Context;
import android.util.Log;

import com.example.signlanguageapplication.Database.SignRecognitionDatabase;
import com.example.signlanguageapplication.Model.Sign;
import com.example.signlanguageapplication.Model.Video;

import java.io.IOException;
import java.util.concurrent.ExecutorService;

public class DictionaryDataService {
    private static final String TAG = "DictionaryDataService";
    private final Context context;
    private final SignRecognitionDatabase database;
    private final ExecutorService executor;

    // Google Drive folder URL - you would need to implement logic to access this
    private static final String DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1099UhspjgRThC8x8RpUH8dIf64dZddMi";

    public DictionaryDataService(Context context) {
        this.context = context;
        this.database = SignRecognitionDatabase.getInstance(context);
        this.executor = SignRecognitionDatabase.getDatabaseWriteExecutor();
    }

    public void loadDictionaryData() {
        executor.execute(() -> {
            try {
                // Check if we already have data
                if (hasDictionaryData()) {
                    Log.d(TAG, "Dictionary data already loaded");
                    return;
                }

                // For initial implementation, we'll download and load from assets folder
                // In a production app, you would implement Google Drive API integration
                loadFromAssets();

            } catch (Exception e) {
                Log.e(TAG, "Error loading dictionary data", e);
            }
        });
    }

    private boolean hasDictionaryData() {
        // Check if we have any signs in the database
        return database.signDao().getSignCount() > 0;
    }

    private void loadFromAssets() {
        try {
            // List all folders in the assets/dictionary directory
            String[] signFolders = context.getAssets().list("dictionary");

            if (signFolders != null) {
                for (String folder : signFolders) {
                    // Each folder is a sign
                    String signName = folder;

                    // Insert sign into database
                    Sign sign = new Sign(signName, "");
                    long signId = database.signDao().insert(sign);

                    // List all videos in this sign folder
                    String[] videos = context.getAssets().list("dictionary/" + folder);
                    if (videos != null) {
                        for (String videoFile : videos) {
                            // Create video entry
                            String videoPath = "file:///android_asset/dictionary/" + folder + "/" + videoFile;
                            String videoTitle = videoFile.replace(".mp4", "");

                            Video video = new Video((int) signId, videoPath, videoTitle);
                            database.videoDao().insert(video);
                        }
                    }
                }
            }
        } catch (IOException e) {
            Log.e(TAG, "Error loading from assets", e);
        }
    }

    // In a real implementation, you would add methods to:
    // 1. Download videos from Google Drive
    // 2. Cache videos locally
    // 3. Update database with new signs and videos
}