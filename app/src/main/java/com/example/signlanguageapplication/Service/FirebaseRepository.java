package com.example.signlanguageapplication.Service;

import android.app.Application;
import android.util.Log;

import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.signlanguageapplication.Database.SignRecognitionDatabase;
import com.example.signlanguageapplication.DAO.SignDao;
import com.example.signlanguageapplication.DAO.VideoDao;
import com.example.signlanguageapplication.Model.Sign;
import com.example.signlanguageapplication.Model.Video;
import com.google.firebase.database.DataSnapshot;
import com.google.firebase.database.DatabaseError;
import com.google.firebase.database.DatabaseReference;
import com.google.firebase.database.FirebaseDatabase;
import com.google.firebase.database.ValueEventListener;
import com.google.firebase.firestore.FirebaseFirestore;
import com.google.firebase.firestore.QueryDocumentSnapshot;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

public class FirebaseRepository {
    private final DatabaseReference signsRef;
    private final FirebaseFirestore firestore;
    private final SignRecognitionDatabase localDatabase;
    private final MutableLiveData<List<Sign>> allSigns = new MutableLiveData<>();
    private final MutableLiveData<List<Video>> videos = new MutableLiveData<>();
    private final MutableLiveData<Boolean> isLoading = new MutableLiveData<>(false);
    private final MutableLiveData<String> errorMessage = new MutableLiveData<>();
    private final Application application;

    public FirebaseRepository(Application application) {
        this.application = application;
        this.signsRef = FirebaseDatabase.getInstance().getReference("signs");
        this.firestore = FirebaseFirestore.getInstance();
        this.localDatabase = SignRecognitionDatabase.getInstance(application);
        syncWithFirebase();
    }

    public LiveData<List<Sign>> getAllSigns() {
        return allSigns;
    }

    public LiveData<List<Sign>> getFavoriteSigns() {
        return localDatabase.signDao().getFavoriteSigns();
    }

    public LiveData<List<Sign>> searchSigns(String query) {
        MutableLiveData<List<Sign>> searchResults = new MutableLiveData<>();
        Log.d("FirebaseRepository", "Searching for query: " + query);
        signsRef.orderByChild("name").startAt(query).endAt(query + "\uf8ff")
                .addListenerForSingleValueEvent(new ValueEventListener() {
                    @Override
                    public void onDataChange(DataSnapshot snapshot) {
                        List<Sign> signs = new ArrayList<>();
                        for (DataSnapshot data : snapshot.getChildren()) {
                            Sign sign = data.getValue(Sign.class);
                            if (sign != null) {
                                signs.add(sign);
                            }
                        }
                        Log.d("FirebaseRepository", "Found " + signs.size() + " signs for query: " + query);
                        for (Sign sign : signs) {
                            Log.d("FirebaseRepository", "Sign: " + sign.getName() + ", ID: " + sign.getId());
                        }
                        searchResults.setValue(signs);
                    }

                    @Override
                    public void onCancelled(DatabaseError error) {
                        Log.e("FirebaseRepository", "Search cancelled: " + error.getMessage());
                        errorMessage.setValue(error.getMessage());
                    }
                });
        return searchResults;
    }

    public LiveData<List<Video>> getVideosForSign(int signId, String signName) {
        MutableLiveData<List<Video>> videoLiveData = new MutableLiveData<>();
        firestore.collection("videos")
                .whereEqualTo("sign_name", signName) // Query theo trường sign_name
                .get()
                .addOnSuccessListener(querySnapshot -> {
                    List<Video> videos = new ArrayList<>();
                    for (QueryDocumentSnapshot document : querySnapshot) {
                        String base64 = document.getString("video_base64");
                        String filename = document.getString("filename");
                        if (base64 != null && filename != null) {
                            Video video = new Video(signId, base64, filename); // Tạo đối tượng Video
                            videos.add(video);
                        }
                    }
                    if (videos.isEmpty()) {
                        errorMessage.setValue("No videos found for sign: " + signName);
                    }
                    videoLiveData.setValue(videos);
                })
                .addOnFailureListener(e -> {
                    errorMessage.setValue("Error fetching videos: " + e.getMessage());
                    videoLiveData.setValue(new ArrayList<>());
                });
        return videoLiveData;
    }

    public LiveData<Boolean> getIsLoading() {
        return isLoading;
    }

    public LiveData<String> getErrorMessage() {
        return errorMessage;
    }

    public void syncWithFirebase() {
        isLoading.setValue(true);
        signsRef.addValueEventListener(new ValueEventListener() {
            @Override
            public void onDataChange(DataSnapshot snapshot) {
                List<Sign> signs = new ArrayList<>();
                for (DataSnapshot data : snapshot.getChildren()) {
                    Sign sign = data.getValue(Sign.class);
                    if (sign != null) {
                        signs.add(sign);
                        SignRecognitionDatabase.getDatabaseWriteExecutor().execute(() -> {
                            Sign existingSign = localDatabase.signDao().getSignByName(sign.getName());
                            if (existingSign == null) {
                                localDatabase.signDao().insert(sign); // Chèn nếu chưa tồn tại
                            } else {
                                // Cập nhật nếu cần
                                existingSign.setDescription(sign.getDescription());
                                existingSign.setBookmarked(sign.isBookmarked());
                                localDatabase.signDao().update(existingSign);
                            }
                        });
                    }
                }
                allSigns.setValue(signs);
                isLoading.setValue(false);
            }

            @Override
            public void onCancelled(DatabaseError error) {
                errorMessage.setValue(error.getMessage());
                isLoading.setValue(false);
            }
        });
    }

    public void toggleFavorite(int signId, boolean isBookmarked) {
        signsRef.child(String.valueOf(signId)).child("bookmarked").setValue(isBookmarked);
        SignRecognitionDatabase.getDatabaseWriteExecutor().execute(() -> {
            localDatabase.signDao().updateBookmark(signId, isBookmarked);
        });
    }
}