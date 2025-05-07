package com.example.signlanguageapplication.View;

import android.os.AsyncTask;
import android.os.Bundle;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.Menu;
import android.view.MenuInflater;
import android.view.MenuItem;
import android.view.View;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.appcompat.widget.SearchView;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.Observer;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.example.signlanguageapplication.Database.SignRecognitionDatabase;
import com.example.signlanguageapplication.Model.SignRecognitionResult;
import com.example.signlanguageapplication.R;
import com.example.signlanguageapplication.ViewModel.SignRecognitionAdapter;
import com.example.signlanguageapplication.ViewModel.SignRecognitionViewModel;
import com.example.signlanguageapplication.databinding.FragmentSignRecognitionBinding;

import java.util.ArrayList;
import java.util.List;

public class SignRecognitionFragment extends Fragment {

    private FragmentSignRecognitionBinding binding;
    private SignRecognitionAdapter adapter;
    private SignRecognitionViewModel viewModel;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentSignRecognitionBinding.inflate(inflater, container, false);
        setHasOptionsMenu(true);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        setupRecyclerView();
        setupViewModel();
        setupToggleButton();
        loadDataFromDatabase();
        debugDatabase(); // Log database data
    }

    private void setupRecyclerView() {
        adapter = new SignRecognitionAdapter();
        binding.rvSignResult.setAdapter(adapter);
        binding.rvSignResult.setLayoutManager(new LinearLayoutManager(requireContext()));
    }

    private void setupViewModel() {
        viewModel = new ViewModelProvider(this).get(SignRecognitionViewModel.class);
        viewModel.connectToServer();

        viewModel.getSignList().observe(getViewLifecycleOwner(), new Observer<ArrayList<SignRecognitionResult>>() {
            @Override
            public void onChanged(ArrayList<SignRecognitionResult> signRecognitionResults) {
                Log.d("DEBUG", "Catch update event");
                adapter.updateSignList(signRecognitionResults);
                Log.d("DEBUG", "Size: " + signRecognitionResults.size());
                if (signRecognitionResults.isEmpty()) {
                    binding.curText.setText("");
                } else {
                    String latestSignName = signRecognitionResults.get(0).getSignName();
                    binding.curText.setText(latestSignName);
                }
                binding.rvSignResult.scrollToPosition(0);
            }
        });
    }

    private void setupToggleButton() {
        binding.toggleSound.setOnCheckedChangeListener((buttonView, isChecked) -> {
            viewModel.setAutoSpeakEnabled(isChecked);
        });
    }

    private void loadDataFromDatabase() {
        AsyncTask.execute(() -> {
            ArrayList<SignRecognitionResult> newList = (ArrayList<SignRecognitionResult>)
                    SignRecognitionDatabase.getInstance(requireContext())
                            .signRecognitionDao()
                            .getAllSignResults();
            Log.d("Database", "Loaded " + newList.size() + " results from database");
            viewModel.setSignList(newList);
        });
    }

    private void debugDatabase() {
        SignRecognitionDatabase.getDatabaseWriteExecutor().execute(() -> {
            List<SignRecognitionResult> results = SignRecognitionDatabase.getInstance(requireContext())
                    .signRecognitionDao()
                    .getAllSignResults();
            Log.d("DatabaseDebug", "Database size: " + results.size());
            for (SignRecognitionResult result : results) {
                Log.d("DatabaseDebug", "Result: " + result.toString());
            }
        });
    }

    @Override
    public void onCreateOptionsMenu(@NonNull Menu menu, @NonNull MenuInflater inflater) {
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

        super.onCreateOptionsMenu(menu, inflater);
    }

    private void filterList(String query) {
        ArrayList<SignRecognitionResult> currentList = viewModel.getSignList().getValue();
        if (currentList == null) {
            adapter.updateSignList(new ArrayList<>());
            return;
        }

        ArrayList<SignRecognitionResult> filteredList = new ArrayList<>();
        for (SignRecognitionResult item : currentList) {
            if (item.getSignName().toLowerCase().contains(query.toLowerCase())) {
                filteredList.add(item);
            }
        }
        adapter.updateSignList(filteredList);
    }

    @Override
    public void onResume() {
        super.onResume();
        loadDataFromDatabase(); // Reload from database when fragment is resumed
        debugDatabase(); // Debug database data
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}