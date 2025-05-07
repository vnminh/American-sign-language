package com.example.signlanguageapplication.Database;

import android.content.Context;
import androidx.room.Database;
import androidx.room.Room;
import androidx.room.RoomDatabase;
import androidx.sqlite.db.SupportSQLiteDatabase;

import com.example.signlanguageapplication.DAO.SignDao;
import com.example.signlanguageapplication.DAO.SignRecognitionDao;
import com.example.signlanguageapplication.DAO.VideoDao;
import com.example.signlanguageapplication.Model.Sign;
import com.example.signlanguageapplication.Model.SignRecognitionResult;
import com.example.signlanguageapplication.Model.Video;

import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;

@Database(entities = {SignRecognitionResult.class, Sign.class, Video.class}, version = 2, exportSchema = false)
public abstract class SignRecognitionDatabase extends RoomDatabase {
    public abstract SignRecognitionDao signRecognitionDao();
    public abstract SignDao signDao();
    public abstract VideoDao videoDao();

    private static volatile SignRecognitionDatabase INSTANCE;
    private static final ExecutorService databaseWriteExecutor = Executors.newFixedThreadPool(4);

    public static SignRecognitionDatabase getInstance(Context context) {
        if (INSTANCE == null) {
            synchronized (SignRecognitionDatabase.class) {
                if (INSTANCE == null) {
                    INSTANCE = Room.databaseBuilder(
                                    context.getApplicationContext(),
                                    SignRecognitionDatabase.class,
                                    "sign_recognition_db")
                            .fallbackToDestructiveMigration()
                            .build();
                }
            }
        }
        return INSTANCE;
    }

    public static ExecutorService getDatabaseWriteExecutor() {
        return databaseWriteExecutor;
    }
}