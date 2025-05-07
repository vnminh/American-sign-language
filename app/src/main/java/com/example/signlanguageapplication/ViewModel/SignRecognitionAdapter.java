package com.example.signlanguageapplication.ViewModel;

import android.content.Context;
import android.os.AsyncTask;
import android.util.Log;
import android.view.LayoutInflater;
import android.view.ViewGroup;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.RecyclerView;

import com.example.signlanguageapplication.Database.SignRecognitionDatabase;
import com.example.signlanguageapplication.Model.SignRecognitionResult;
import com.example.signlanguageapplication.databinding.ItemSignRecognitionBinding;

import java.util.ArrayList;

public class SignRecognitionAdapter extends RecyclerView.Adapter<SignRecognitionAdapter.SignViewHolder> {
    private ArrayList<SignRecognitionResult> signList = new ArrayList<>();

    @NonNull
    @Override
    public SignViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        ItemSignRecognitionBinding binding = ItemSignRecognitionBinding.inflate(LayoutInflater.from(parent.getContext()), parent, false);
        return new SignViewHolder(binding);
    }

    @Override
    public void onBindViewHolder(@NonNull SignViewHolder holder, int position) {
        holder.bind(signList.get(position));
    }

    @Override
    public int getItemCount() {
        return signList.size();
    }

    public void updateSignList(ArrayList<SignRecognitionResult> newList) {
        DiffUtil.DiffResult diffResult = DiffUtil.calculateDiff(new SignDiffCallback(signList, newList));
        signList = newList;
        diffResult.dispatchUpdatesTo(this);
    }

    public void addSignResult(SignRecognitionResult result) {
        this.signList.add(0, result);
        notifyItemInserted(0);
    }

    // Định nghĩa ViewHodler để render item
    static class SignViewHolder extends RecyclerView.ViewHolder {
        private final ItemSignRecognitionBinding binding;

        public SignViewHolder(ItemSignRecognitionBinding binding) {
            super(binding.getRoot());
            this.binding = binding;
        }

        public void bind(SignRecognitionResult result) {
            this.binding.tvSignName.setText(result.getSignName());
            this.binding.tvTimestamp.setText(result.getTimestamp());
        }
    }

    static class SignDiffCallback extends DiffUtil.Callback {
        private final ArrayList<SignRecognitionResult> oldList;
        private final ArrayList<SignRecognitionResult> newList;

        public SignDiffCallback(ArrayList<SignRecognitionResult> oldList, ArrayList<SignRecognitionResult> newList) {
            this.oldList = oldList;
            this.newList = newList;
        }

        @Override
        public int getOldListSize() {
            return oldList.size();
        }

        @Override
        public int getNewListSize() {
            return newList.size();
        }

        @Override
        public boolean areItemsTheSame(int oldItemPosition, int newItemPosition) {
            return oldList.get(oldItemPosition).getTimestamp().equals(newList.get(newItemPosition).getTimestamp());
        }

        @Override
        public boolean areContentsTheSame(int oldItemPosition, int newItemPosition) {
            return oldList.get(oldItemPosition).equals(newList.get(newItemPosition));
        }
    }
}