package com.example.signlanguageapplication.View;

import android.os.Bundle;
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
import androidx.lifecycle.ViewModelProvider;
import androidx.navigation.Navigation;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.example.signlanguageapplication.R;
import com.example.signlanguageapplication.ViewModel.DictionaryAdapter;
import com.example.signlanguageapplication.ViewModel.DictionaryViewModel;
import com.example.signlanguageapplication.databinding.FragmentDictionaryBinding;

public class DictionaryFragment extends Fragment {

    private FragmentDictionaryBinding binding;
    private DictionaryViewModel viewModel;
    private DictionaryAdapter adapter;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentDictionaryBinding.inflate(inflater, container, false);
        setHasOptionsMenu(true);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        setupRecyclerView();
        setupViewModel();
    }

    private void setupRecyclerView() {
        adapter = new DictionaryAdapter(sign -> {
            // Navigate to sign detail when sign is clicked
            Bundle bundle = new Bundle();
            bundle.putInt("signId", sign.getId());
            bundle.putString("signName", sign.getName());
            Navigation.findNavController(requireView()).navigate(
                    R.id.action_dictionaryFragment_to_signDetailFragment, bundle);
        });

        binding.rvDictionary.setAdapter(adapter);
        binding.rvDictionary.setLayoutManager(new LinearLayoutManager(requireContext()));
    }

    private void setupViewModel() {
        viewModel = new ViewModelProvider(requireActivity()).get(DictionaryViewModel.class);

        // Load initial data
        viewModel.getAllSigns().observe(getViewLifecycleOwner(), signs -> {
            adapter.submitList(signs);
            updateEmptyState(signs.isEmpty());
        });

        // Make sure dictionary data is loaded
        viewModel.loadDictionaryData();
    }

    private void updateEmptyState(boolean isEmpty) {
        if (isEmpty) {
            binding.emptyStateLayout.setVisibility(View.VISIBLE);
            binding.rvDictionary.setVisibility(View.GONE);
        } else {
            binding.emptyStateLayout.setVisibility(View.GONE);
            binding.rvDictionary.setVisibility(View.VISIBLE);
        }
    }

    @Override
    public void onCreateOptionsMenu(@NonNull Menu menu, @NonNull MenuInflater inflater) {
        inflater.inflate(R.menu.menu_toolbar, menu);

        MenuItem searchItem = menu.findItem(R.id.action_search);
        SearchView searchView = (SearchView) searchItem.getActionView();

        searchView.setOnQueryTextListener(new SearchView.OnQueryTextListener() {
            @Override
            public boolean onQueryTextSubmit(String query) {
                performSearch(query);
                return true;
            }

            @Override
            public boolean onQueryTextChange(String newText) {
                performSearch(newText);
                return true;
            }
        });

        super.onCreateOptionsMenu(menu, inflater);
    }

    private void performSearch(String query) {
        viewModel.searchSigns(query); // Gọi phương thức searchSigns từ ViewModel
        viewModel.getAllSigns().observe(getViewLifecycleOwner(), signs -> {
            adapter.submitList(signs);
            updateEmptyState(signs.isEmpty());
        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        binding = null;
    }
}