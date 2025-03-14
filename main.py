from scipy.io.wavfile import read as wav_r
from scipy import fft
import scipy as sc
import matplotlib.pyplot as plt
import numpy as np


def find_note(frequency):
    notes = [' A', ' A#', ' B', 'C', 'C#', ' D', ' D#', ' E', ' F', ' F#', ' G', ' G#']

    note_number = round(12 * np.log2(frequency / 440) + 49)
    note = notes[(note_number - 1) % len(notes)]
    octave = (note_number + 8) // len(notes)
    if octave > 0:
        return note, octave
    else:
        return "none", "none"


def find_and_label_peaks(signal, freqs):
    peaks = sc.signal.find_peaks(np.round(signal), threshold=1.5, prominence=15, width=2, distance=20)
    peaks = peaks[0]

    peak_freq = []
    peak_value = []
    peak_note = []

    for peak_i in peaks:
        new_peak = signal[peak_i]
        peak_value.append(new_peak)

        new_freq = freqs[peak_i]
        peak_freq.append(new_freq)

        new_note, new_octave = find_note(new_freq)
        peak_note.append(new_note + str(new_octave))

        print(round(new_freq, 2), "&", round(10 * np.log10(new_peak), 2), "&", new_note, "&", new_octave , "\\\\")
    return peak_freq, peak_value, peak_note


wave = wav_r("chord.wav")
sample_rate = wave[0]
print(sample_rate)
wave = wave[1]
# wave = wave[16000:]
duration = float(wave.size / sample_rate)

fig1 = plt.figure("Waveform")
plt.plot(np.arange(0, duration, 1 / sample_rate), wave, 'b',
         linewidth=1.5)
# plt.axis([0, duration, -1.2, 1.2])
plt.xlabel("t(s)")
plt.ylabel("Amplitude")

tfreq = fft.fftfreq(wave.size, 1 / sample_rate)
twave = fft.fft(wave)
psd = 2 / wave.size * np.abs(twave[round(twave.size / 2):])
freqs = np.abs(tfreq[round(twave.size / 2):])

peak_freq, peak_value, peak_note = find_and_label_peaks(psd, freqs)

fig1_5 = plt.figure("Full Power Density Spectrum no peaks")
plt.plot(freqs, 10 * np.log10(psd), 'b', linewidth=1)
plt.xlabel("f(Hz)")
plt.ylabel("SPL(dB)")
plt.axis([16, 4000, -30, 30])
plt.xscale('log')

fig2 = plt.figure("Full Power Density Spectrum")
ax = fig2.add_subplot(111)
plt.plot(freqs, 10 * np.log10(psd), 'b', linewidth=1)
plt.scatter(peak_freq, 10 * np.log10(peak_value), c='k')
plt.xlabel("f(Hz)")
plt.ylabel("SPL(dB)")
plt.axis([16, 4000, -30, 30])
plt.xscale('log')

for i in range(len(peak_freq)):
    ax.text(peak_freq[i], 10 * np.log10(peak_value[i]), "  " + peak_note[i])

fig3 = plt.figure("PDS_Notes")
ax = fig3.add_subplot(111)

plt.plot(freqs, 10 * np.log10(psd), 'b', linewidth=1)
plt.xticks(list(plt.xticks()[0]) + peak_freq)
plt.scatter(peak_freq, 10 * np.log10(peak_value), c='k')
plt.xlabel("f(Hz)")
plt.ylabel("SPL(dB)")
plt.axis([60, 350, -2, 30])

for i in range(len(peak_freq)):
    if peak_freq[i] < 330 and peak_freq[i] > 75:
        ax.text(peak_freq[i], 10 * np.log10(peak_value[i]), "  " + peak_note[i])

plt.show()
