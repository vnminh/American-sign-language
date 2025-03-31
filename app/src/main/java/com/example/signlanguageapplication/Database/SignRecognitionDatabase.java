package com.example.signlanguageapplication.Database;

import android.content.Context;
import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;

import com.example.signlanguageapplication.DAO.SignRecognitionDao;
import com.example.signlanguageapplication.Model.SignRecognitionResult;

@Database(entities = {SignRecognitionResult.class}, version = 1, exportSchema = false)
public abstract class SignRecognitionDatabase extends RoomDatabase {
    public abstract SignRecognitionDao signRecognitionDao();

    private static volatile SignRecognitionDatabase INSTANCE;

    public static SignRecognitionDatabase getInstance(Context context) {
        if (INSTANCE == null) {
            INSTANCE = Room.databaseBuilder(context,SignRecognitionDatabase.class,"sign_recognition_db").build();
        }
        return INSTANCE;
    }
}
