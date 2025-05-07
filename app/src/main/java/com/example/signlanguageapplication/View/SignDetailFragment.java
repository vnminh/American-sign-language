package com.example.signlanguageapplication.View;

import android.os.Bundle;
import android.util.Base64;
import android.net.Uri;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ImageButton;
import android.widget.TextView;
import android.widget.Toast;

import androidx.annotation.NonNull;
import androidx.annotation.Nullable;
import androidx.fragment.app.Fragment;
import androidx.lifecycle.ViewModelProvider;
import androidx.recyclerview.widget.LinearLayoutManager;

import com.example.signlanguageapplication.Model.Video;
import com.example.signlanguageapplication.ViewModel.DictionaryViewModel;
import com.example.signlanguageapplication.ViewModel.VideoAdapter;
import com.example.signlanguageapplication.R;
import com.example.signlanguageapplication.databinding.FragmentSignDetailBinding;
import com.google.android.exoplayer2.ExoPlayer;
import com.google.android.exoplayer2.MediaItem;
import com.google.android.exoplayer2.PlaybackException;
import com.google.android.exoplayer2.Player;

import java.io.File;
import java.io.FileOutputStream;

public class SignDetailFragment extends Fragment {

    private FragmentSignDetailBinding binding;
    private DictionaryViewModel viewModel;
    private VideoAdapter adapter;
    private int signId;
    private String signName;
    private ExoPlayer exoPlayer;
    private float[] playbackSpeeds = {0.25f, 0.5f, 0.75f, 1.0f, 1.25f, 1.5f, 1.75f, 2.0f};
    private int currentSpeedIndex = 3;

    @Override
    public View onCreateView(LayoutInflater inflater, ViewGroup container, Bundle savedInstanceState) {
        binding = FragmentSignDetailBinding.inflate(inflater, container, false);
        return binding.getRoot();
    }

    @Override
    public void onViewCreated(@NonNull View view, @Nullable Bundle savedInstanceState) {
        super.onViewCreated(view, savedInstanceState);

        if (getArguments() != null) {
            signId = getArguments().getInt("signId", -1);
            signName = getArguments().getString("signName", "");
        }

        if (signId == -1) {
            requireActivity().onBackPressed();
            return;
        }

        binding.tvSignTitle.setText(signName);
        setupRecyclerView();
        setupViewModel();
    }

    private void setupRecyclerView() {
        adapter = new VideoAdapter(video -> playVideo(video));
        binding.rvVideos.setAdapter(adapter);
        binding.rvVideos.setLayoutManager(new LinearLayoutManager(requireContext()));
    }

    private void setupViewModel() {
        viewModel = new ViewModelProvider(requireActivity()).get(DictionaryViewModel.class);
        viewModel.getVideosForSign(signId, signName).observe(getViewLifecycleOwner(), videos -> {
            adapter.submitList(videos);
            updateEmptyState(videos.isEmpty());
        });
    }

    private void updateEmptyState(boolean isEmpty) {
        if (isEmpty) {
            binding.emptyStateLayout.setVisibility(View.VISIBLE);
            binding.rvVideos.setVisibility(View.GONE);
        } else {
            binding.emptyStateLayout.setVisibility(View.GONE);
            binding.rvVideos.setVisibility(View.VISIBLE);
        }
    }

    private void playVideo(Video video) {
        binding.videoPlayerLayout.setVisibility(View.VISIBLE);
        binding.progressBar.setVisibility(View.VISIBLE);

        try {
            byte[] videoBytes = Base64.decode(video.getVideoBase64(), Base64.DEFAULT);
            File tempFile = new File(requireContext().getCacheDir(), video.getFilename());
            FileOutputStream fos = new FileOutputStream(tempFile);
            fos.write(videoBytes);
            fos.close();

            if (exoPlayer == null) {
                exoPlayer = new ExoPlayer.Builder(requireContext()).build();
                binding.videoView.setPlayer(exoPlayer);

                // Inflate control layout
                LayoutInflater inflater = LayoutInflater.from(requireContext());
                View controlView = inflater.inflate(R.layout.exo_player_control_view, binding.controlLayout, true);

                // Get control elements
                ImageButton replayButton = controlView.findViewById(R.id.exo_replay);
                ImageButton rewindButton = controlView.findViewById(R.id.exo_rew);
                ImageButton playPauseButton = controlView.findViewById(R.id.exo_play_pause);
                ImageButton fastForwardButton = controlView.findViewById(R.id.exo_ffwd);
                ImageButton speedUpButton = controlView.findViewById(R.id.exo_speed_up);
                ImageButton speedDownButton = controlView.findViewById(R.id.exo_speed_down);
                TextView speedText = controlView.findViewById(R.id.exo_speed_text);

                // Replay button
                replayButton.setOnClickListener(v -> {
                    if (exoPlayer != null) {
                        exoPlayer.seekTo(0);
                        exoPlayer.setPlayWhenReady(true);
                    }
                });

                // Rewind button
                rewindButton.setOnClickListener(v -> {
                    if (exoPlayer != null) {
                        long currentPosition = exoPlayer.getCurrentPosition();
                        exoPlayer.seekTo(Math.max(0, currentPosition - 5000)); // Tua lại 5 giây
                    }
                });

                // Play/Pause button
                playPauseButton.setOnClickListener(v -> {
                    if (exoPlayer != null) {
                        exoPlayer.setPlayWhenReady(!exoPlayer.isPlaying());
                    }
                });

                // Fast forward button
                fastForwardButton.setOnClickListener(v -> {
                    if (exoPlayer != null) {
                        long currentPosition = exoPlayer.getCurrentPosition();
                        long duration = exoPlayer.getDuration();
                        exoPlayer.seekTo(Math.min(duration, currentPosition + 5000)); // Tua nhanh 5 giây
                    }
                });

                // Speed up button
                speedUpButton.setOnClickListener(v -> {
                    if (exoPlayer != null) {
                        currentSpeedIndex = Math.min(currentSpeedIndex + 1, playbackSpeeds.length - 1);
                        exoPlayer.setPlaybackSpeed(playbackSpeeds[currentSpeedIndex]);
                        speedText.setText(playbackSpeeds[currentSpeedIndex] + "x");
                        Toast.makeText(requireContext(), "Tốc độ: " + playbackSpeeds[currentSpeedIndex] + "x", Toast.LENGTH_SHORT).show();
                    }
                });

                // Speed down button
                speedDownButton.setOnClickListener(v -> {
                    if (exoPlayer != null) {
                        currentSpeedIndex = Math.max(currentSpeedIndex - 1, 0);
                        exoPlayer.setPlaybackSpeed(playbackSpeeds[currentSpeedIndex]);
                        speedText.setText(playbackSpeeds[currentSpeedIndex] + "x");
                        Toast.makeText(requireContext(), "Tốc độ: " + playbackSpeeds[currentSpeedIndex] + "x", Toast.LENGTH_SHORT).show();
                    }
                });

                // Update play/pause icon dynamically
                exoPlayer.addListener(new Player.Listener() {
                    @Override
                    public void onPlayWhenReadyChanged(boolean playWhenReady, int reason) {
                        if (playPauseButton != null) {
                            playPauseButton.setImageResource(playWhenReady ? android.R.drawable.ic_media_pause : android.R.drawable.ic_media_play);
                        }
                    }
                });
            }

            Uri videoUri = Uri.fromFile(tempFile);
            MediaItem mediaItem = MediaItem.fromUri(videoUri);
            exoPlayer.setMediaItem(mediaItem);
            exoPlayer.prepare();
            exoPlayer.setPlaybackSpeed(playbackSpeeds[currentSpeedIndex]);
            exoPlayer.setPlayWhenReady(true);

            exoPlayer.addListener(new Player.Listener() {
                @Override
                public void onPlaybackStateChanged(int playbackState) {
                    if (playbackState == Player.STATE_BUFFERING) {
                        binding.progressBar.setVisibility(View.VISIBLE);
                    } else if (playbackState == Player.STATE_READY || playbackState == Player.STATE_ENDED) {
                        binding.progressBar.setVisibility(View.GONE);
                    }
                }

                @Override
                public void onPlayerError(PlaybackException error) {
                    binding.progressBar.setVisibility(View.GONE);
                    Toast.makeText(requireContext(), "Lỗi phát video: " + error.getMessage(), Toast.LENGTH_SHORT).show();
                    binding.videoPlayerLayout.setVisibility(View.GONE);
                }
            });

            TextView speedText = binding.controlLayout.findViewById(R.id.exo_speed_text);
            speedText.setText(playbackSpeeds[currentSpeedIndex] + "x");

        } catch (Exception e) {
            binding.progressBar.setVisibility(View.GONE);
            Toast.makeText(requireContext(), "Lỗi giải mã video: " + e.getMessage(), Toast.LENGTH_SHORT).show();
            binding.videoPlayerLayout.setVisibility(View.GONE);
        }

        binding.btnClosePlayer.setOnClickListener(v -> {
            if (exoPlayer != null) {
                exoPlayer.stop();
                exoPlayer.release();
                exoPlayer = null;
            }
            binding.videoPlayerLayout.setVisibility(View.GONE);
        });
    }

    @Override
    public void onDestroyView() {
        super.onDestroyView();
        if (exoPlayer != null) {
            exoPlayer.release();
            exoPlayer = null;
        }
        if (binding != null && binding.videoView != null) {
            binding.videoView.setPlayer(null);
        }
        binding = null;
    }
}