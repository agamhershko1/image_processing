# Note: All spectrogram plotting calls have been commented out.

import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as sig
from scipy.io import wavfile


def create_watermarks(audio_path) -> None:
    # Load the audio file
    sample_rate, audio = wavfile.read(audio_path)

    # plot_spectrogram(audio, sample_rate, title="Original Spectrogram")

    # Create a "bad" watermark at 10kHz
    apply_frequency_watermark(
        audio, 'Task 1/bad_watermark.wav', sample_rate, 10000)

    # Create a "good" watermark at 18kHz
    apply_frequency_watermark(
        audio, 'Task 1/good_watermark.wav', sample_rate, 18000)


def apply_frequency_watermark(audio, output_path, sample_rate, watermark_freq):
    f, t_stft, Zxx = sig.stft(audio, fs=sample_rate)

    # Find the closest frequency bin to the desired watermark frequency
    freq_index = np.argmin(np.abs(f - watermark_freq))

    # Increase the magnitude of the selected frequency bin
    Zxx[freq_index, :] += 200

    watermarked_audio = sig.istft(Zxx, fs=sample_rate)[1]

    # plot_spectrogram(watermarked_audio, sample_rate, title="Spectrogram with watermark at " +
    #                                                        str(watermark_freq) + " Hz")

    # Save the watermarked audio to the specified output path
    wavfile.write(output_path, sample_rate, watermarked_audio.astype(audio.dtype))


def classify_by_watermarks():
    """
    Classifies watermarked audio files based on their watermark frequency patterns.

    Returns:
    list: Groups of audio files classified by watermark period with their period time
    """
    watermark_groups = []

    for i in range(9):
        audio_path = f'Task 2/{i}_watermarked.wav'
        sample_rate, audio = wavfile.read(audio_path)

        # plot_spectrogram(audio, sample_rate, title="Spectrogram of audio 2." + str(i))

        f, t_stft, sxx = sig.stft(audio, fs=sample_rate)

        # Target frequency (20kHz) for watermark detection
        target_freq = 20000
        freq_idx = np.argmin(np.abs(f - target_freq))

        # Extract the magnitude of the target frequency
        magnitude = np.abs(sxx[freq_idx, :])
        threshold = np.max(magnitude) * 0.11  # Define a threshold for detecting watermark
        times_with_target_freq = t_stft[magnitude > threshold]

        period_time = int(get_period_time(times_with_target_freq))

        # Group audio files with similar watermark periods
        added_to_group = False
        for group in watermark_groups:
            if abs(period_time - group[1]) < 0.1 * group[1]:  # 10% threshold for similarity
                group[0].append(i)
                # Update the group's average period using a weighted moving average
                group[1] = (len(group[0]) - 1) / len(group[0]) * group[1] + 1 / len(group[0]) * period_time
                added_to_group = True
                break

        if not added_to_group:
            watermark_groups.append([[i], int(period_time)])

    return watermark_groups


def get_period_time(times):
    """
    Calculates the average period between significant time intervals.
    """
    differences = []

    for i in range(len(times) - 2):
        diff = times[i + 2] - times[i]
        if diff > 1:  # Ignore short intervals
            differences.append(diff)

    return np.mean(differences)


def speed_up():
    """"
    Plotting relevant spectrograms (for each audio file).
    """
    for i in range(1, 3):
        audio_path = 'Task 3/task3_watermarked_method' + str(i) + '.wav'
        sample_rate, audio = wavfile.read(audio_path)

        plot_spectrogram(audio, sample_rate, title="Spectrogram number" + str(i))


def plot_spectrogram(audio, sample_rate, title="Spectrogram"):
    # Generate the spectrogram
    frequencies, seg_times, sxx = sig.spectrogram(audio, sample_rate)

    # Add a small epsilon value to avoid log of zero
    epsilon = 1e-10
    sxx = np.maximum(sxx, epsilon)  # Ensure all values are >= epsilon

    # Plot the spectrogram using a logarithmic scale
    plt.pcolormesh(seg_times, frequencies, 10 * np.log10(sxx), shading='auto')
    plt.ylabel('Frequency [Hz]')
    plt.xlabel('Time [sec]')
    plt.title(title)
    plt.colorbar(label='Intensity [dB]')
    plt.show()


if __name__ == '__main__':
    # Part 1
    create_watermarks('Task 1/task1.wav')

    # Part 2
    print(classify_by_watermarks())

    # Part 3
    speed_up()
