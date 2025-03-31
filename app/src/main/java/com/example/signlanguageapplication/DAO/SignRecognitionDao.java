package com.example.signlanguageapplication.DAO;


import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import com.example.signlanguageapplication.Model.SignRecognitionResult;
import java.util.List;

@Dao
public interface SignRecognitionDao {
    @Insert
    void insert(SignRecognitionResult result);

    @Query("SELECT * FROM sign_results ORDER BY id DESC")
    List<SignRecognitionResult> getAllSignResults();

    @Query("DELETE FROM sign_results")
    void clearDatabase();
}
