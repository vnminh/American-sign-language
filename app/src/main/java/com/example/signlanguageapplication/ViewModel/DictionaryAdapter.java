package com.example.signlanguageapplication.ViewModel;

import android.view.LayoutInflater;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.example.signlanguageapplication.Model.Sign;
import com.example.signlanguageapplication.databinding.ItemDictionaryBinding;

public class DictionaryAdapter extends ListAdapter<Sign, DictionaryAdapter.SignViewHolder> {

    private final OnSignClickListener listener;

    public interface OnSignClickListener {
        void onSignClick(Sign sign);
    }

    public DictionaryAdapter(OnSignClickListener listener) {
        super(new SignDiffCallback());
        this.listener = listener;
    }

    @NonNull
    @Override
    public SignViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        ItemDictionaryBinding binding = ItemDictionaryBinding.inflate(
                LayoutInflater.from(parent.getContext()), parent, false);
        return new SignViewHolder(binding);
    }

    @Override
    public void onBindViewHolder(@NonNull SignViewHolder holder, int position) {
        holder.bind(getItem(position), listener);
    }

    static class SignViewHolder extends RecyclerView.ViewHolder {
        private final ItemDictionaryBinding binding;

        public SignViewHolder(ItemDictionaryBinding binding) {
            super(binding.getRoot());
            this.binding = binding;
        }

        public void bind(Sign sign, OnSignClickListener listener) {
            binding.tvSignName.setText(sign.getName());
            if (sign.getDescription() != null && !sign.getDescription().isEmpty()) {
                binding.tvSignDescription.setText(sign.getDescription());
                binding.tvSignDescription.setVisibility(android.view.View.VISIBLE);
            } else {
                binding.tvSignDescription.setVisibility(android.view.View.GONE);
            }

            // Set the bookmark/favorite icon if needed
            // binding.ivBookmark.setVisibility(sign.isBookmarked() ? View.VISIBLE : View.GONE);

            binding.getRoot().setOnClickListener(v -> listener.onSignClick(sign));
        }
    }

    static class SignDiffCallback extends DiffUtil.ItemCallback<Sign> {
        @Override
        public boolean areItemsTheSame(@NonNull Sign oldItem, @NonNull Sign newItem) {
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull Sign oldItem, @NonNull Sign newItem) {
            return oldItem.getName().equals(newItem.getName()) &&
                    (oldItem.getDescription() == null ? newItem.getDescription() == null :
                            oldItem.getDescription().equals(newItem.getDescription())) &&
                    oldItem.isBookmarked() == newItem.isBookmarked();
        }
    }
}