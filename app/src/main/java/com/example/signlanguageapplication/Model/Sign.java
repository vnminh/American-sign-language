package com.example.signlanguageapplication.Model;

import androidx.room.Entity;
import androidx.room.Ignore;
import androidx.room.PrimaryKey;

@Entity(tableName = "signs")
public class Sign {
    @PrimaryKey(autoGenerate = true)
    private int id;
    private String name;
    private String description;
    private boolean bookmarked;

    // Non-parameterized constructor for Firebase
    @Ignore
    public Sign() {}

    public Sign(String name, String description) {
        this.name = name;
        this.description = description;
        this.bookmarked = false;
    }

    public int getId() { return id; }
    public void setId(int id) { this.id = id; }

    public String getName() { return name; }
    public void setName(String name) { this.name = name; }

    // For compatibility with SignAdapter that uses getWord()
    public String getWord() { return name; }

    public String getDescription() { return description; }
    public void setDescription(String description) { this.description = description; }

    public boolean isBookmarked() { return bookmarked; }
    public void setBookmarked(boolean bookmarked) { this.bookmarked = bookmarked; }

}