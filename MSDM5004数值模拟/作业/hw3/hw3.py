import numpy as np
import matplotlib.pyplot as plt
# 1
# (a)
piano = np.loadtxt('C:/Users/张铭韬/Desktop/学业/港科大/MSDM5004数值模拟/作业/hw3/piano.txt')
# Plot the piano
plt.figure(figsize=(12, 12))
plt.plot(piano)
plt.title('piano')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

dft_piano = np.abs(np.fft.fft(piano))[:10000]
freq_piano = np.fft.fftfreq(len(piano), d=1/44100)[:10000]

# Plot the magnitudes
plt.figure(figsize=(12, 12))
plt.plot(freq_piano,dft_piano)
plt.title('Magnitudes(absolute value) of Piano DFT Coefficients')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()

trumpet = np.loadtxt('C:/Users/张铭韬/Desktop/学业/港科大/MSDM5004数值模拟/作业/hw3/trumpet.txt')
# Plot the trumpet
plt.figure(figsize=(12, 12))
plt.plot(trumpet)
plt.title('trumpet')
plt.xlabel('Time')
plt.ylabel('Amplitude')
plt.show()

dft_trumpet = np.abs(np.fft.fft(trumpet))[:10000]
freq_trumpet = np.fft.fftfreq(len(trumpet), d=1/44100)[:10000]

# Plot the magnitudes
plt.figure(figsize=(12, 12))
plt.plot(freq_trumpet, dft_trumpet)
plt.title('Magnitudes(absolute value) of Trumpet DFT Coefficients')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.show()
# Piano sounds often have a faster amplitude decay, resulting in a sharp drop.
# The sound of a trumpet typically exhibits a slower amplitude decay than a piano, resulting in a more gradual downward trend.
# Trumpet sounds are higher in frequency and have greater overall amplitude.


# (b)
import math
def freq_to_note(freq):
    notes = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']
    note_number = 12 * math.log2(freq / 440) + 49  
    note_number = round(note_number)
    note = (note_number - 1 ) % len(notes)
    note = notes[note]
    octave = (note_number + 8 ) // len(notes)
    return note, octave

def calculate_frequency(signal, sample_rate):
    spectrum = np.fft.fft(signal)
    magnitude = np.abs(spectrum)
    freq_axis = np.fft.fftfreq(len(signal), 1/sample_rate)
    max_index = np.argmax(magnitude)
    frequency = freq_axis[max_index]
    return abs(frequency)

note_piano = freq_to_note(calculate_frequency(piano, 44100))
print(note_piano)
note_trumpet = freq_to_note(calculate_frequency(trumpet, 44100))
print(note_trumpet)
# which means the note C


# 2



















