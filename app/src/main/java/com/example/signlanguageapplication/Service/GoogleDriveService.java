package com.example.signlanguageapplication.Service;

import android.content.Context;
import android.net.Uri;
import android.os.Environment;
import android.util.Log;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;
import java.net.HttpURLConnection;
import java.net.URL;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class GoogleDriveService {
    private static final String TAG = "GoogleDriveService";
    private final Context context;
    private final ExecutorService executor;
    private final File cacheDir;

    // Callback interfaces
    public interface OnFolderListFetchedListener {
        void onFoldersFetched(List<DriveFolder> folders);
        void onError(String error);
    }

    public interface OnFileDownloadListener {
        void onDownloadComplete(File file);
        void onDownloadFailed(String error);
    }

    public GoogleDriveService(Context context) {
        this.context = context;
        this.executor = Executors.newFixedThreadPool(4);

        // Create cache directory
        this.cacheDir = new File(context.getExternalFilesDir(Environment.DIRECTORY_MOVIES), "asl_videos");
        if (!cacheDir.exists()) {
            cacheDir.mkdirs();
        }
    }

    public void downloadFile(String fileId, OnFileDownloadListener listener) {
        executor.execute(() -> {
            File cachedFile = new File(cacheDir, fileId + ".mp4");

            // Check if file is already cached
            if (cachedFile.exists()) {
                listener.onDownloadComplete(cachedFile);
                return;
            }

            // Not cached, download from Drive
            try {
                // Create direct download URL
                String downloadUrl = "https://drive.google.com/uc?export=download&id=" + fileId;
                URL url = new URL(downloadUrl);
                HttpURLConnection connection = (HttpURLConnection) url.openConnection();

                // Download file
                try (InputStream input = connection.getInputStream();
                     FileOutputStream output = new FileOutputStream(cachedFile)) {

                    byte[] buffer = new byte[4096];
                    int bytesRead;

                    while ((bytesRead = input.read(buffer)) != -1) {
                        output.write(buffer, 0, bytesRead);
                    }
                }

                listener.onDownloadComplete(cachedFile);
            } catch (IOException e) {
                Log.e(TAG, "Error downloading file", e);
                listener.onDownloadFailed("Failed to download: " + e.getMessage());
            }
        });
    }

    public Uri getLocalUriForFile(String fileId) {
        File cachedFile = new File(cacheDir, fileId + ".mp4");
        if (cachedFile.exists()) {
            return Uri.fromFile(cachedFile);
        }
        return null;
    }

    public boolean isFileCached(String fileId) {
        File cachedFile = new File(cacheDir, fileId + ".mp4");
        return cachedFile.exists();
    }

    public static class DriveFolder {
        private final String id;
        private final String name;

        public DriveFolder(String id, String name) {
            this.id = id;
            this.name = name;
        }

        public String getId() {
            return id;
        }

        public String getName() {
            return name;
        }
    }

    public static class DriveFile {
        private final String id;
        private final String name;

        public DriveFile(String id, String name) {
            this.id = id;
            this.name = name;
        }

        public String getId() {
            return id;
        }

        public String getName() {
            return name;
        }
    }
}