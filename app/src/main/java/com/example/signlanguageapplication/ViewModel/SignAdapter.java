package com.example.signlanguageapplication.ViewModel;

import android.view.LayoutInflater;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.RecyclerView;

import com.example.signlanguageapplication.Model.Sign;
import com.example.signlanguageapplication.R;
import com.example.signlanguageapplication.databinding.ItemSignBinding;

import java.util.ArrayList;
import java.util.List;

public class SignAdapter extends RecyclerView.Adapter<SignAdapter.SignViewHolder> {
    private List<Sign> signList = new ArrayList<>();
    private final OnSignClickListener onSignClickListener;
    private final OnFavoriteClickListener onFavoriteClickListener;

    public interface OnSignClickListener {
        void onSignClick(Sign sign);
    }

    public interface OnFavoriteClickListener {
        void onFavoriteClick(Sign sign);
    }

    public SignAdapter(OnSignClickListener onSignClickListener, OnFavoriteClickListener onFavoriteClickListener) {
        this.onSignClickListener = onSignClickListener;
        this.onFavoriteClickListener = onFavoriteClickListener;
    }

    @NonNull
    @Override
    public SignViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        ItemSignBinding binding = ItemSignBinding.inflate(
                LayoutInflater.from(parent.getContext()), parent, false);
        return new SignViewHolder(binding);
    }

    @Override
    public void onBindViewHolder(@NonNull SignViewHolder holder, int position) {
        Sign sign = signList.get(position);
        holder.bind(sign);
    }

    @Override
    public int getItemCount() {
        return signList.size();
    }

    public void updateList(List<Sign> newList) {
        this.signList = new ArrayList<>(newList);
        notifyDataSetChanged();
    }

    class SignViewHolder extends RecyclerView.ViewHolder {
        private final ItemSignBinding binding;

        public SignViewHolder(ItemSignBinding binding) {
            super(binding.getRoot());
            this.binding = binding;

            // Set click listeners
            binding.getRoot().setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (position != RecyclerView.NO_POSITION && onSignClickListener != null) {
                    onSignClickListener.onSignClick(signList.get(position));
                }
            });

            binding.ibFavorite.setOnClickListener(v -> {
                int position = getAdapterPosition();
                if (position != RecyclerView.NO_POSITION && onFavoriteClickListener != null) {
                    onFavoriteClickListener.onFavoriteClick(signList.get(position));
                }
            });
        }

        void bind(Sign sign) {
            binding.tvSignWord.setText(sign.getWord());
            binding.ibFavorite.setImageResource(
                    sign.isBookmarked() ?
                            R.drawable.ic_bookmark_filled :
                            R.drawable.ic_bookmark_outline);
        }
    }
}