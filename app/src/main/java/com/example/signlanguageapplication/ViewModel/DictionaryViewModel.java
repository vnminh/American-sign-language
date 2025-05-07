package com.example.signlanguageapplication.ViewModel;

import android.app.Application;

import androidx.annotation.NonNull;
import androidx.lifecycle.AndroidViewModel;
import androidx.lifecycle.LiveData;
import androidx.lifecycle.MutableLiveData;

import com.example.signlanguageapplication.Model.Sign;
import com.example.signlanguageapplication.Model.Video;
import com.example.signlanguageapplication.Service.FirebaseRepository;

import java.util.List;

public class DictionaryViewModel extends AndroidViewModel {
    private final FirebaseRepository repository;
    private final MutableLiveData<String> searchQuery = new MutableLiveData<>("");
    private final MutableLiveData<Boolean> showFavoritesOnly = new MutableLiveData<>(false);

    public DictionaryViewModel(@NonNull Application application) {
        super(application);
        repository = new FirebaseRepository(application);
    }

    public LiveData<List<Sign>> getAllSigns() {
        if (showFavoritesOnly.getValue() != null && showFavoritesOnly.getValue()) {
            return repository.getFavoriteSigns();
        } else if (searchQuery.getValue() != null && !searchQuery.getValue().isEmpty()) {
            return repository.searchSigns(searchQuery.getValue());
        } else {
            return repository.getAllSigns();
        }
    }

    // Cập nhật phương thức để truyền cả signId và signName
    public LiveData<List<Video>> getVideosForSign(int signId, String signName) {
        return repository.getVideosForSign(signId, signName);
    }

    public LiveData<Boolean> getIsLoading() {
        return repository.getIsLoading();
    }

    public LiveData<String> getErrorMessage() {
        return repository.getErrorMessage();
    }

    public void loadDictionaryData() {
        repository.syncWithFirebase();
    }

    public void searchSigns(String query) {
        this.searchQuery.setValue(query);
    }

    public void toggleFavorites(boolean showFavoritesOnly) {
        this.showFavoritesOnly.setValue(showFavoritesOnly);
    }

    public void toggleFavoriteStatus(Sign sign) {
        repository.toggleFavorite(sign.getId(), !sign.isBookmarked());
    }
}