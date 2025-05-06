package com.example.signlanguageapplication;


import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;

import androidx.activity.EdgeToEdge;
import androidx.appcompat.app.AppCompatActivity;
import androidx.appcompat.widget.SearchView;
import androidx.appcompat.widget.Toolbar;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.example.signlanguageapplication.DAO.SignRecognitionDao;
import com.example.signlanguageapplication.Database.SignRecognitionDatabase;
import com.example.signlanguageapplication.Model.SignRecognitionResult;
import com.example.signlanguageapplication.ViewModel.SignRecognitionAdapter;
import com.example.signlanguageapplication.ViewModel.SignRecognitionViewModel;
import com.example.signlanguageapplication.databinding.ActivityMainBinding;

import java.util.ArrayList;

public class MainActivity extends AppCompatActivity {

    private ActivityMainBinding binding;
    private SignRecognitionAdapter adapter;
    private SignRecognitionViewModel viewModel;

    private SignRecognitionDatabase database;

    private SignRecognitionDao recognitionDao;


    @Override
    public boolean onCreateOptionsMenu(Menu menu) {
        MenuInflater inflater = getMenuInflater();
        inflater.inflate(R.menu.menu_toolbar, menu);

        MenuItem searchItem = menu.findItem(R.id.action_search);
        SearchView searchView = (SearchView) searchItem.getActionView();

        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {
                filterList(query);
                return true;
            }

            @Override
            public boolean onQueryTextChange(String newText) {
                filterList(newText);
                return true;
            }
        });

        return true;
    }

    private void filterList(String query) {
        ArrayList<SignRecognitionResult> filteredList = new ArrayList<>();

        for (SignRecognitionResult item : viewModel.getSignList().getValue()) {
            if (item.getSignName().toLowerCase().contains(query.toLowerCase())) {
                filteredList.add(item);
            }
        }

        adapter.updateSignList(filteredList);
    }




    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        // Khởi tạo viewBinding activity_main
        binding = ActivityMainBinding.inflate(getLayoutInflater());
        setContentView(binding.getRoot());

        Toolbar toolbar = findViewById(R.id.toolbar);
        setSupportActionBar(toolbar);

        // Khởi tạo RecyclerView
        adapter = new SignRecognitionAdapter();
        adapter.addSignResult(new SignRecognitionResult("test","hehe"));
        binding.rvSignResult.setAdapter(adapter);
        binding.rvSignResult.setLayoutManager(new LinearLayoutManager(this));

        // Khởi tạo ViewModel & Kết nối với server
        viewModel = new SignRecognitionViewModel(getApplicationContext());
        viewModel.connectToServer();

        viewModel.getSignList().observe(this, new Observer<ArrayList<SignRecognitionResult>>() {
            @Override
            public void onChanged(ArrayList<SignRecognitionResult> signRecognitionResults) {
                Log.d("DEBUG","Catch update event");
                adapter.updateSignList(signRecognitionResults);
                Log.d("DEBUG","Size: " + signRecognitionResults.size());
                if(signRecognitionResults.isEmpty()) {
                    binding.curText.setText("");
                }
                else {
                    String latestSignName = signRecognitionResults.get(0).getSignName();
                    binding.curText.setText(latestSignName);
                }
                binding.rvSignResult.scrollToPosition(0); // Scroll to top after updating list
            }
        });

        binding.toggleSound.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setAutoSpeakEnabled(isChecked);
        });

        AsyncTask.execute(new Runnable() {
            @Override
            public void run() {
//                SignRecognitionDatabase.getInstance(getApplicationContext()).signRecognitionDao().clearDatabase();
                ArrayList<SignRecognitionResult> newList = (ArrayList<SignRecognitionResult>) SignRecognitionDatabase.getInstance(getApplicationContext()).signRecognitionDao().getAllSignResults();
                viewModel.setSignList(newList);
            }
        });




    }



}


