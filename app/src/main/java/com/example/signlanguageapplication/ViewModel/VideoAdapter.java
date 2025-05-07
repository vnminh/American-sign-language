package com.example.signlanguageapplication.ViewModel;

import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.TextView;

import androidx.annotation.NonNull;
import androidx.recyclerview.widget.DiffUtil;
import androidx.recyclerview.widget.ListAdapter;
import androidx.recyclerview.widget.RecyclerView;

import com.example.signlanguageapplication.Model.Video;
import com.example.signlanguageapplication.R;

public class VideoAdapter extends ListAdapter<Video, VideoAdapter.VideoViewHolder> {

    private final OnVideoClickListener onVideoClickListener;

    public VideoAdapter(OnVideoClickListener onVideoClickListener) {
        super(DIFF_CALLBACK);
        this.onVideoClickListener = onVideoClickListener;
    }

    private static final DiffUtil.ItemCallback<Video> DIFF_CALLBACK = new DiffUtil.ItemCallback<Video>() {
        @Override
        public boolean areItemsTheSame(@NonNull Video oldItem, @NonNull Video newItem) {
            // So sánh dựa trên signId hoặc id
            return oldItem.getId() == newItem.getId();
        }

        @Override
        public boolean areContentsTheSame(@NonNull Video oldItem, @NonNull Video newItem) {
            // So sánh nội dung dựa trên videoBase64 và filename
            return oldItem.getVideoBase64().equals(newItem.getVideoBase64()) &&
                    oldItem.getFilename().equals(newItem.getFilename());
        }
    };

    @NonNull
    @Override
    public VideoViewHolder onCreateViewHolder(@NonNull ViewGroup parent, int viewType) {
        View view = LayoutInflater.from(parent.getContext())
                .inflate(R.layout.item_video, parent, false);
        return new VideoViewHolder(view);
    }

    @Override
    public void onBindViewHolder(@NonNull VideoViewHolder holder, int position) {
        Video video = getItem(position);
        holder.bind(video);
    }

    public class VideoViewHolder extends RecyclerView.ViewHolder {
        private final TextView tvVideoTitle;

        public VideoViewHolder(@NonNull View itemView) {
            super(itemView);
            tvVideoTitle = itemView.findViewById(R.id.tv_video_title);

            itemView.setOnClickListener(v -> {
                if (onVideoClickListener != null) {
                    int position = getAdapterPosition();
                    if (position != RecyclerView.NO_POSITION) {
                        onVideoClickListener.onVideoClick(getItem(position));
                    }
                }
            });
        }

        public void bind(Video video) {
            // Sử dụng filename làm tiêu đề
            tvVideoTitle.setText(video.getFilename());
        }
    }

    public interface OnVideoClickListener {
        void onVideoClick(Video video);
    }
}