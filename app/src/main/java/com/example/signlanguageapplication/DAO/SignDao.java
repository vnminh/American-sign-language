package com.example.signlanguageapplication.DAO;

import androidx.lifecycle.LiveData;
import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;
import androidx.room.Update;

import com.example.signlanguageapplication.Model.Sign;

import java.util.List;

@Dao
public interface SignDao {
    @Insert
    long insert(Sign sign);

    @Update
    void update(Sign sign);

    @Query("SELECT * FROM signs ORDER BY name ASC")
    LiveData<List<Sign>> getAllSigns();

    @Query("SELECT * FROM signs WHERE name LIKE :query ORDER BY name ASC")
    LiveData<List<Sign>> searchSigns(String query);

    @Query("SELECT * FROM signs WHERE bookmarked = 1 ORDER BY name ASC")
    LiveData<List<Sign>> getFavoriteSigns();

    @Query("SELECT * FROM signs WHERE id = :id")
    LiveData<Sign> getSignById(int id);

    @Query("SELECT * FROM signs WHERE name = :name LIMIT 1")
    Sign getSignByName(String name);

    @Query("SELECT COUNT(*) FROM signs")
    int getSignCount();

    @Query("UPDATE signs SET bookmarked = :bookmarked WHERE id = :id")
    void updateBookmark(int id, boolean bookmarked);
}