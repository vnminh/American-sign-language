/*
 * @Author: Phuc Nguyen nguyenhuuphuc22052004@gmail.com
 * @Date: 2025-04-16 22:12:34
 * @LastEditors: Phuc Nguyen nguyenhuuphuc22052004@gmail.com
 * @LastEditTime: 2025-04-17 11:15:56
 * @FilePath: \SignLanguageApplication\app\src\main\java\com\example\signlanguageapplication\DAO\VideoDao.java
 * @Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
 */
package com.example.signlanguageapplication.DAO;

import androidx.lifecycle.LiveData;
import androidx.room.Dao;
import androidx.room.Insert;
import androidx.room.Query;

import com.example.signlanguageapplication.Model.Video;

import java.util.List;

@Dao
public interface VideoDao {
    @Insert
    void insert(Video video);

    @Query("SELECT * FROM videos WHERE signId = :signId")
    LiveData<List<Video>> getVideosForSign(int signId);
}