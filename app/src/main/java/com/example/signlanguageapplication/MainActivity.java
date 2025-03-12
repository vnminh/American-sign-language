package com.example.signlanguageapplication;


import android.os.Bundle;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.example.signlanguageapplication.Model.SignRecognitionResult;
import com.example.signlanguageapplication.ViewModel.SignRecognitionAdapter;
import com.example.signlanguageapplication.ViewModel.SignRecognitionViewModel;
import com.example.signlanguageapplication.databinding.ActivityMainBinding;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private SignRecognitionAdapter adapter;
    private SignRecognitionViewModel viewModel;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Khởi tạo viewBinding activity_main
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        // Khởi tạo RecyclerView
        adapter = new SignRecognitionAdapter();
        binding.rvSignResult.setAdapter(adapter);
        binding.rvSignResult.setLayoutManager(new LinearLayoutManager(this));

        // Khởi tạo ViewModel & Kết nối với server
        viewModel = new ViewModelProvider(this).get(SignRecognitionViewModel.class);
        viewModel.connectToServer();

        viewModel.getSignList().observe(this, adapter::updateSignList);
    }
}