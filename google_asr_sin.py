"""Test Google Cloud ASR offerings on the Speech in Noise (SPIN) test."""

import datetime
import json
import re
from typing import List, Dict, Optional, Set, Tuple, Union

from absl import app
from absl import flags

import dataclasses
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
from scipy.optimize import curve_fit
import numpy as np
import os

import fsspec

from google.cloud import speech_v2

from google.api_core.client_options import ClientOptions
from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech

from google.cloud.speech_v1.types.cloud_speech import RecognizeResponse
# Supported Languages:
# https://cloud.google.com/speech-to-text/v2/docs/speech-to-text-supported-languages


################### Speech Recognition Class (easier API) ######################

class RecognitionEngine(object):
  """A class that provides a nicer interface to Google's Cloud
  text-to-speech API.

  Here are some useful links:
    https://cloud.google.com/speech-to-text/docs/speech-to-text-supported-languages
  """
  def __init__(self):
    self._client = None
    self._parent = None

  def CreateSpeechClient(self,
                         gcp_project,
                         model='default_long',
                         ):
    """Acquires the appropriate authentication and creates a Cloud Speech stub.

    The model name is needed because we connect to a different server if the
    model is 'chirp'.

    Returns:
      a Cloud Speech stub.
    """
    self._model = model
    self._project = gcp_project
    self._spoken_punct = False
    self._auto_punct = False


    if model == 'chirp':
      chirp_endpoint = 'us-central1-speech.googleapis.com'
      client_options = ClientOptions(api_endpoint=chirp_endpoint)
      self._location = 'us-central1'
    else:
      client_options = ClientOptions()
      self._location = 'global'
    self._client = SpeechClient(client_options=client_options)

  def ListModels(self, gcp_project: str):
    if self._client is None:
      self.CreateSpeechClient(gcp_project)
    parent = f'projects/{self._project}/locations/{self._location}'
    request = speech_v2.ListRecognizersRequest(parent=parent)
    return self._client.ListModels(request)

  def ListRecognizers(self, gcp_project: str):
    if self._client is None:
      self.CreateSpeechClient(gcp_project)
    parent = f'projects/{self._project}/locations/{self._location}'
    request = speech_v2.ListRecognizersRequest(parent=parent)
    # print(f'ListRecognizers request is: {request}')
    return self._client.list_recognizers(request)

  def CreateRecognizer(self,
                       with_timings=False,
                       locale: str = 'en-US',
                       # gcp_project: str,
                       # recognizer_id: str,
                       # debug=False
                       ):
    # https://cloud.google.com/speech-to-text/v2/docs/medical-models
    if self._model == 'medical_conversation':
      self._spoken_punct = False
      self._auto_punct = True
    elif self._model == 'medical_dictation':
      self._spoken_punct = True
      self._auto_punct = True
    else:
      self._spoken_punct = False
      self._auto_punct = False

    self._recognizer_config = cloud_speech.RecognitionConfig(
        auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=[locale],
        model=self._model,
        features = speech_v2.RecognitionFeatures(
                  enable_word_time_offsets = with_timings,
                  enable_automatic_punctuation = self._auto_punct,
                  enable_spoken_punctuation = self._spoken_punct,
              ),
    )

  def RecognizeFile(self,
                    audio_file_path: str,
                    with_timings=False,
                    debug=False) -> cloud_speech.RecognizeResponse:
    """Recognize the speech from a file.
    Returns: 
    https://cloud.google.com/python/docs/reference/speech/latest/google.cloud.speech_v1.types.RecognizeResponse
    
    Note: Unless the file ends in .wav, the file is read in, and the entire
    contents, including the binary header, are passed to the recognizer as a
    16kHz audio waveform."""
    if audio_file_path.endswith('.wav'):
      with fsspec.open(audio_file_path, 'rb') as fp:
        audio_fs, audio_data = wavfile.read(fp)
        return self.RecognizeWaveform(audio_data, audio_fs,
                                      with_timings=with_timings)

    recognizer_name = (f'projects/{self._project}/locations/'
                       f'{self._location}/recognizers/_')
    # Create the request we'd like to send
    request = cloud_speech.RecognizeRequest(
        recognizer = recognizer_name,
        config = self._recognizer_config,
        content = self.ReadAudioFile(audio_file_path)
    )
    # Send the request
    if debug:
      print(request)
    response = self._client.recognize(request)
    return response

  def RecognizeWaveform(self,
                        waveform: Union[bytes, np.ndarray],
                        sample_rate: int = 16000,
                        with_timings=False,
                        debug=False) -> RecognizeResponse:
    """Recognize the speech from a waveform."""
    if isinstance(waveform, np.ndarray):
      waveform = waveform.astype(np.int16).tobytes()

    recognizer_name = (f'projects/{self._project}/locations/'
                       f'{self._location}/recognizers/_')
    # Create the request we'd like to send
    self._recognizer_config = cloud_speech.RecognitionConfig(
        explicit_decoding_config = cloud_speech.ExplicitDecodingConfig(
            # Change these based on the encoding of the audio
            # See the encoding documentation on how to do this.
            # https://cloud.google.com/speech-to-text/v2/docs/encoding
            encoding = 'LINEAR16',
            sample_rate_hertz = sample_rate,
            audio_channel_count = 1,
        ),
        # auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
        language_codes=['en-US'],
        model=self._model,
        features = speech_v2.RecognitionFeatures(
                  enable_word_time_offsets = with_timings,
                  enable_automatic_punctuation = self._auto_punct,
                  enable_spoken_punctuation = self._spoken_punct,
              ),
    )

    request = cloud_speech.RecognizeRequest(
        recognizer = recognizer_name,
        config = self._recognizer_config,
        content = waveform
    )
    if debug:
      print(request)
    # Send the request

    response = self._client.recognize(request)
    return response

  def ReadAudioFile(self, audio_file_path: str):
    # if audio_file_path[0] != '/':
    #   PREFIX = '/google_src/files/head/depot/'
    #   audio_file_path = os.path.join(PREFIX, audio_file_path)
    with fsspec.open(audio_file_path, 'rb') as audio_file:
      audio_data = audio_file.read()
    return audio_data

####### Utilities to Parse Google Recognition Results ########
@dataclasses.dataclass
class RecogResult:
  word: str
  start_time: float
  end_time: float


def parse_time(time_proto) -> float:
  # print(f'Time_proto is a {type(time_proto)}')
  # return time_proto.seconds + time_proto.nanos/1e9
  return time_proto.total_seconds()

def parse_transcript(response:
                     cloud_speech.RecognizeResponse) -> List[RecogResult]:
  """Parse the results from the Cloud ASR engine and return a simple list
  of words and times.  This is for the entire (60s) utterance."""
  words = []
  for a_result in response.results:
    try:
      # For reasons I don't understand sometimes a results is missing the
      # alternatives
      l = len(a_result.alternatives) > 0
      if not l:
        continue
    except: # pylint: disable=bare-except
      continue
    for word in a_result.alternatives[0].words:
      # print(f'Processing: {word}')
      start_time = parse_time(word.start_offset)
      end_time = parse_time(word.end_offset)
      recog_result = RecogResult(word.word.lower(), start_time, end_time)
      words.append(recog_result)
    words.append(RecogResult('.', end_time, end_time))
    # print(words[-1])
  return words

def print_all_sentences(results: cloud_speech.RecognizeResponse):
  for r in results:
    if r.alternatives:
      print(r.alternatives[0].transcript)
    else:
      print('No alternatives')


####### Utilities to prepare original SPIN waveforms ########

def generate_ffmpeg_cmds():
  """Generate the FFMPEG commands to downsample and rename the
  QuickSIN files. The Google drive data from Matt has these files:
*   34 Sep List 11.aif - Stereo utterances: clean sentences on the left,
      constant amplitude babble noise on the right
*   34 Sep List 11_sentence.wav - Mono clean sentences
*   34 Sep List 11_babble.wav - Mono babble
*   List 11.aif - Mono mixed test sentences, with the SNR stepping
      down after each sentence.

  """
  for i in range(1, 13):
    input_name = f'{23+i} Sep List {i}_sentence.wav'
    output_name = f'QuickSIN22/Clean List {i}.wav'
    print(f'ffmpeg -i "{input_name}" -ar 22050 "{output_name}"')

    input_name = f'List {i}.aif'
    output_name = f'QuickSIN22/Babble List {i}.wav'
    print(f'ffmpeg -i "{input_name}" -ar 22050 "{output_name}"')

    print()


################## Organize SPIN recogntion results #######################

# A list of lists.  Each (final) list is a list of recognition results (words
# and times). Then a list of these "sentence" lists.
SpinFileTranscripts = List[List[RecogResult]]

def recognize_all_spin(all_wavs: List[str],
                       asr_engine: RecognitionEngine,
                       debug=False) -> SpinFileTranscripts:
  """Recognize some SPiN sentences using the specified ASR engine.
  Return a list of the transcription results.  Each recognition result is
  a list of alternatives, all in RecogResult format. This is used for both 
  clean and noisy utterances.
  """
  all_results = []
  for f in all_wavs:
    if 'Calibration' in f:
      continue
    pretty_file_name = os.path.basename(f)
    if debug:
      print('Recognizing', pretty_file_name)
    resp = asr_engine.RecognizeFile(f, with_timings=True, debug=debug)
    if debug:
      print(f'{pretty_file_name}:',)
      for result in resp.results:
        if result.alternatives:
          print(f'   {result.alternatives[0].transcript}')
        else:
          print('.   ** Empty ASR Result **')
    recog_results = parse_transcript(resp)
    all_results.append(recog_results)
  return all_results


def find_sentence_boundaries(
    spin_truth_names: List[str],  # File names with clean speech.
    sentence_boundary_graph: str = '') -> Tuple[List[int], np.ndarray]:
  """Figure out the inter-sentence boundaries of each 
  sentence in all lists. Do this by summing the absolute value of each
  waveform, filter this to get an envelope, then look for the
  minimums.

  Return a list of sample numbers indicating the midpoint between sentences.
  """
  # Figure out the maximum length
  max_len = 0
  for i in range(12):
    with fsspec.open(spin_truth_names[i], 'rb') as fp:
      audio_fs, audio_data = wavfile.read(fp)
      max_len = max(max_len, len(audio_data))
  # Now sum the absolute value of each of the 12 waveforms.
  all_audio = np.zeros(max_len, float)
  for i in range(12):
    with fsspec.open(spin_truth_names[i], 'rb') as fp:
      _, audio_data = wavfile.read(fp)
      all_audio[:len(audio_data)] = (all_audio[:len(audio_data)] +
                                     np.abs(audio_data))

  # Now filter this signal to snmooth it.
  b, a = signal.butter(4, 0.00005)
  envelope = signal.filtfilt(b, a, all_audio, padlen=150)
  envelope = signal.filtfilt(b, a, envelope, padlen=150)

  def find_min(y, start, stop):
    start_sample = int(start)
    end_sample = int(stop)
    i = np.argmin(y[start_sample:end_sample]) + start_sample
    return float(i)

  # Look for the minimum in each approximate range.
  splits = np.array([0.2, 0.3, 0.5, 0.7, 0.9, 1.1])*1e6  # This is in samples.
  breaks = [0]
  for i in range(5):
    breaks.append(find_min(envelope, splits[i], splits[i+1])/audio_fs)
  breaks.append(max_len/audio_fs)

  # Plot the results
  if sentence_boundary_graph:
    plt.clf()
    plt.plot(np.arange(len(all_audio))/float(audio_fs), all_audio)
    plt.xlabel('Time (s)')
    plt.ylabel('Average Audio Level')
    plt.title('Sentence Boundaries from Average Amplitude')
    current_axis = plt.axis()
    for b in breaks:
      plt.plot([b, b], current_axis[2:], '--')
    plt.savefig(sentence_boundary_graph)

  return breaks, all_audio

######################## QuickSIN Ground Truth ################################

# Pages 96-97 of this PhD thesis:
# Suzanne E. Sklaney, Binaural sound field presentation of the QuickSIN:
# Equivalncy across lists and signal-to-noise ratios.
# https://etda.libraries.psu.edu/files/final_submissions/5788

key_word_list = """
L 0 S 0  white silk jacket any shoes
L 0 S 1  child crawled into dense grass
L 0 S 2  Footprints showed path took beach
L 0 S 3  event near edge fresh air
L 0 S 4  band Steel 3/three inches/in wide
L 0 S 5  weight package seen high scale

L 1 S 0  tear/Tara thin sheet yellow pad
L 1 S 1  cruise Waters Sleek yacht fun
L 1 S 2  streak color down left Edge
L 1 S 3  done before boy see it
L 1 S 4  Crouch before jump miss mark
L 1 S 5  square peg settle round hole

L 2 S 0  pitch straw through door stable
L 2 S 1  sink thing which pile dishes
L 2 S 2  post no bills office wall
L 2 S 3  dimes showered down all sides
L 2 S 4  pick card slip under pack/pact
L 2 S 5  store jammed before sale start

L 3 S 0  sense smell better than touch
L 3 S 1  picked up dice second roll
L 3 S 2  drop ashes worn/Warren Old rug
L 3 S 3  couch cover Hall drapes blue
L 3 S 4  stems Tall Glasses cracked broke
L 3 S 5  cleats sank deeply soft turf

L 4 S 0  have better than wait Hope
L 4 S 1  screen before fire kept Sparks
L 4 S 2  thick glasses helped read print
L 4 S 3  chair looked strong no bottom
L 4 S 4  told wild Tales/tails frighten him
L 4 S 5  force equal would move Earth

L 5 S 0  leaf drifts along slow spin
L 5 S 1  pencil cut sharp both ends
L 5 S 2  down road way grain farmer
L 5 S 3  best method fix place clips
L 5 S 4  if Mumble your speech lost
L 5 S 5  toad Frog hard tell apart

L 6 S 0  kite dipped swayed/suede stayed aloft
L 6 S 1  beatle/beetle drowned hot June sun/son
L 6 S 2  theft Pearl pin Kept Secret
L 6 S 3  wide grin earned many friends
L 6 S 4  hurdle pit aid long Pole
L 6 S 5  Peep under tent see Clown

L 7 S 0  sun came light Eastern sky
L 7 S 1  stale smell old beer lingers
L 7 S 2  desk firm on shaky floor
L 7 S 3  list names carved around base
L 7 S 4  news struct/struck out Restless Minds
L 7 S 5  Sand drifts over sill house

L 8 S 0  take shelter tent keep still
L 8 S 1  Little Tales/tails they tell false
L 8 S 2  press pedal with left foot
L 8 S 3  black trunk fell from Landing
L 8 S 4  cheap clothes flashy don't last
L 8 S 5  night alarm roused/roust deep sleep

L 9 S 0  dots light betrayed black cat
L 9 S 1  put chart mantle Tack down
L 9 S 2  steady drip worse drenching rain
L 9 S 3  flat pack less luggage space
L 9 S 4  gloss top made unfit read
L 9 S 5  Seven Seals stamped great sheets

L10 S 0  marsh freeze when cold enough
L10 S 1  gray mare walked before colt
L10 S 2  bottles hold four kinds rum
L10 S 3  wheeled/wheled bike past winding road
L10 S 4  throw used paper cup plate
L10 S 5  wall phone ring loud often

L11 S 0  hinge door creaked old age
L11 S 1  bright lanterns Gay dark lawn
L11 S 2  offered proof  form large chart
L11 S 3  their eyelids droop want sleep
L11 S 4  many ways do these things
L11 S 5  we like see clear weather
""".split('\n')


def word_alternatives(words: str,
                      homonyms_dict: Dict[str, Set[str]]) -> Set[str]:
  """Convert a string with words separated by '/' into a set."""
  all_words = words.strip().split('/')
  base_word = all_words[0]
  if base_word in homonyms_dict:
    return set(all_words) | homonyms_dict[base_word]
  return set(all_words)


homonyms = """
  # Add word equivalances, here.
  # Homonyms
  tails/tales
  4/four
  four/for
  maire/mare
  pedal/petal
  wheeled/wield
  sun/son
  marsh/marsue
  their/there
  white/whitesilk
  silk/whitesilk
  roll/role
  drowned/dround
  yacht/yaught
  hall/haul
  # Close enough words.
  # None so far.. we count if an error if even one phoneme is wrong.
"""

def make_homonyms_dictionary(*equivalance_lists: str) -> Dict[str, Set[str]]:
  """Convert a set of speech-recognition equivalances, specified as text, into
  a dictionary of sets of equivalent words.  Each equivalence is specified as
  words on a line, separated by '/'.  Lines that start with '#' are ignored so
  that the choices can be documented.
  """
  result_dict = {}
  for equivalance in equivalance_lists:
    equivalance = equivalance.split('\n')
    equivalance_lines = [line.strip().split('/') for line in equivalance
                         if line.strip() and line.strip()[0] != '#']
    # print(all_sets)
    for a_set in equivalance_lines:
      # print(f'Processing {a_set}')
      w = a_set[0] # The base term
      if w in result_dict:
        raise ValueError(f'Found duplicate key {w}')
      else:
        result_dict[w] = set(a_set[1:])
      # print(f'After {w} dict is {result_dict}')
  return result_dict


def ingest_quicksin_truth(
    word_list: str,
    homonym_dict: Dict[str, Set[str]]) -> Dict[Tuple[int, int],
                                               List[Set[str]]]:
  """Convert the text from the big string above into a set of key words 
  (and alternatives) that describe the expected answers from a SPIN test.

  For each line (which will be entered into a dictionary keyed by list and
  sentence number) create a list of test words, where each test word is stored 
  as a list of alternatives in a set.
  """
  keyword_dict = {}
  for line in word_list:
    line = line.strip().lower()
    if not line: continue
    list_number = int(line[1:3])
    sentence_number = int(line[5:7])
    key_words = line[7:].split(' ')
    key_words = [w for w in key_words if w]
    key_list = [word_alternatives(w, homonym_dict) for w in key_words]
    if len(key_list) != 5:
      print(f'Have too many words in L{list_number} S{sentence_number}:',
            key_list)
    keyword_dict[list_number, sentence_number] = key_list
  return keyword_dict

homonym_list = make_homonyms_dictionary(homonyms)
all_keyword_dict = ingest_quicksin_truth(key_word_list, homonym_list)

######## Recognize the SPIN waveforms and calculate all word timings ##########

@dataclasses.dataclass
class SpinSentence:
  """A structure that describes one SPiN sentence, with the transcript,
  individual words, the sentence start and end time, and the SNR.

  There are six SPiN sentences per list, one per SNR.
  """
  sentence_words: List[str]
  true_word_list: List[Set[str]]  # List of words and their alternatives
  # words: list[str]
  start_time: float
  end_time: float
  snr: float  # This sentence's test SNR


# Organize the clean speech transcripts.  Each 60s wavedform becomes a list of
# recognized sentences.  Return a list of list of sentences.

spin_snrs = (25, 20, 15, 10, 5, 0)

def format_quicksin_truth(
    spin_transcripts: SpinFileTranscripts,  # List of List of RecogResults
    sentence_breaks: List[float],           # Times in seconds
    snr_list: Tuple[float] = spin_snrs) -> List[List[SpinSentence]]:
  """Parse the recognition results and produce a List (of sentences at different
  SNRs).  Return a list of 12 SPIN lists, each list containing the 6 SPIN 
  sentences at the different SNRs.
  """
  assert len(spin_transcripts) > 0
  # assert len(sentence_breaks) == 7  # Nominally 7 except when testing
  # assert len(snr_list) == 6         # Nominally 6 except when testing
  spin_results = []
  # Iterate through the lists (each list contains 6 different sentences)
  print('Sentence breaks are at:', sentence_breaks)
  for list_number, clean_transcript in enumerate(spin_transcripts):
    sentences = []
    for snr_number, snr in enumerate(snr_list):
      sentence_start_time = float(sentence_breaks[snr_number])
      sentence_end_time = float(sentence_breaks[snr_number+1])
      sentence_words = [w for w in clean_transcript
                        if (w.start_time > sentence_start_time and
                            w.end_time < sentence_end_time)]
      assert len(sentence_words) > 0, (f'No words found for list {list_number},'
                                       f' snr #{snr_number} between '
                                       f'{sentence_start_time}s and '
                                       f'{sentence_end_time}s.')
      recognized_words = [w.word for w in sentence_words]
      sentence = SpinSentence(recognized_words,
                              all_keyword_dict[list_number, snr_number],
                              min(*[w.start_time for w in sentence_words]),
                              max(*[w.start_time for w in sentence_words]),
                              snr
                              )
      sentences.append(sentence)
    spin_results.append(sentences)
  return spin_results


def print_spin_ground_truth(truth: List[List[SpinSentence]]):
  for spin_i, spin_list in enumerate(truth):
    print(f'\nQuickSIN list {spin_i}')
    for sentence in spin_list:
      print(f'SNR {sentence.snr} '
            f'from {sentence.start_time}s to {sentence.end_time}s:',
            ' '.join(sentence.sentence_words)
            )


def save_ground_truth(truth: List[List[SpinSentence]], filename: str):
  """Save the QuickSIN ground truth into a JSON file so we don't have
  to compute it again."""
  class GoogleSinEncoder(json.JSONEncoder):
    def default(self, o):
      if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
      elif isinstance(o, set):
        return list(o)
      return super().default(o)

  saved_data = {
    'ground_truth': truth,
    'time': str(datetime.datetime.now()),
  }
  with fsspec.open(filename, 'w') as fp:
    json.dump(saved_data, fp, cls=GoogleSinEncoder)


def load_ground_truth(filename: str) -> List[List[SpinSentence]]:
  """Load the precomputed QuickSIN ground truth from a file.
  
  Args:
    The file from which to read the cached ground truth.

  Return:
    A list of 12 SPIN lists, each list containing the 6 SPIN 
    sentences at the different SNRs.
  """
  with fsspec.open(filename, 'r') as fp:
    saved_data = json.load(fp)
  if isinstance(saved_data, dict):
    truth = saved_data['ground_truth']
  else:
    truth = saved_data  # Old format file
  print(f'Reloading ground truth saved at {saved_data["time"]}')
  assert isinstance(truth, list)
  for i in range(len(truth)):        # Nominally 12, except during testing
    assert isinstance(truth[i], list)
    for s in range(len(truth[i])):   # Nominally 6, except during testing
      truth[i][s] = SpinSentence(**truth[i][s])
      truth[i][s].true_word_list = [set(word_list) for word_list
                                    in truth[i][s].true_word_list]
  return truth


number_re = re.compile(r' (\d+).wav')

def sort_by_list_number(s: str) -> int:
  m = number_re.search(s)
  assert m, f'Could not find list number in {s}'
  return int(m[1])


def compute_quicksin_truth(
    wav_dir: str,
    project_id: str,
    sentence_breaks: Optional[List[float]] = None,
    snr_list: Tuple[float] = spin_snrs,
    sentence_boundary_graph: str = '') -> List[List[SpinSentence]]:
  """Create the ground truth for a SPIN test. 
  Process all the clean speech files to figure out the start and stop of each
  sentence.  Combine with the keyword list to create a list (by QuickSin list) 
  of lists of sentences (one sentence per test SNR).
  """
  spin_file_names = fsspec.open_files(os.path.join(wav_dir, '*.wav'))
  spin_file_names = [f.full_name for f in spin_file_names]
  spin_truth_names = [f for f in spin_file_names if 'Clean' in f]
  assert spin_truth_names, f'Could not find clean speech files in {wav_dir}.'

  # Make sure these are all sorted numerically.
  spin_truth_names.sort(key=sort_by_list_number)
  print(f'Found {len(spin_truth_names)} QuickSIN lists to process.')

  if sentence_breaks is None:
    print('Finding sentence boundaries...')
    sentence_breaks, _ = find_sentence_boundaries(spin_truth_names,
                                                  sentence_boundary_graph)
    print('Sentence breaks are:', sentence_breaks)

  model = 'latest_long'
  print(f'Transcribing the QuickSIN WAV files with {model} model....')
  asr_engine = RecognitionEngine()
  asr_engine.CreateSpeechClient(project_id, model)
  asr_engine.CreateRecognizer(with_timings=True)

  true_transcripts = recognize_all_spin(spin_truth_names, asr_engine)
  print('True transcripts are:')
  for l in range(len(true_transcripts)):
    for s in range(len(true_transcripts[l])):
      print(f'List {l}, Sentence {s}:', true_transcripts[l][s])


  print('Formatting the QuickSIN Ground Truth....')
  spin_ground_truths = format_quicksin_truth(true_transcripts,
                                             sentence_breaks,
                                             snr_list)
  return spin_ground_truths

################ SCORE ALL MODELS IN NOISE  ############################
def words_in_trial(recognized_words: List[RecogResult],
                   start_time: float,  # Seconds
                   end_time: float,    # Seconds
                   tolerance: float = 2.0) -> List[str]:
  """Pick out the words in the babble mixture that fall within time window."""
  start_time -= tolerance
  end_time += tolerance
  # print(recognized_words[0].keys())
  words = [r.word for r in recognized_words
           if r.end_time >= start_time and r.start_time <= end_time]
  # Remove all but word characters (not punctuation)
  words = [re.sub(r'[^\w]', '', word.lower()) for word in words]
  return words


def prettyprint_words_and_alternatives(words_and_alternatives):
  results = []
  for w in words_and_alternatives:
    if isinstance(w, str):
      results.append(w)
    elif isinstance(w, (list, set)):
      results.append('/'.join(list(w)))
    else:
      raise ValueError(f'Unexpected type in {words_and_alternatives}')
  return results


def score_word_list(true_words: List[Set[str]],
                    recognized_words: List[str], max_count=0) -> int:
  """How many of the key words show up in the transcript?
  Args: 
    true_words: a list of tuples, each tuple is a list of words 
      and their alternates
    recognized_words: A list of recognized words to score.
    max_count: Maximum number to return

  Returns:
    The number of correctly (as judged by the true_words list) that
    were recognized.
  """
  score = 0
  missing_words = []
  for words_and_alternates in true_words:
    for word in words_and_alternates:
      found = False
      if word in recognized_words:
        found = True
        break
    if found:
      score += 1
    else:
      missing_words.append(words_and_alternates)
  if max_count:
    score = min(score, max_count)
  if missing_words:
    missing_words = prettyprint_words_and_alternatives(missing_words)
    #pylint: disable=inconsistent-quotes
    print(f'Want {", ".join(missing_words)}, '
          f'from: {", ".join(recognized_words)}')
  return score

def score_all_tests(snrs: List[float],
                    ground_truths: List[List[SpinSentence]],
                    reco_results: List[RecogResult],
                    debug=False) -> np.ndarray:
  num_lists = len(ground_truths)
  num_keywords = 5

  correct_counts = []
  for snr_num, snr in enumerate(snrs):
    correct_count = 0
    for list_num in range(num_lists):
      true_words = ground_truths[list_num][snr_num].true_word_list
      recognized_words = words_in_trial(
        reco_results[list_num],
        ground_truths[list_num][snr_num].start_time,
        ground_truths[list_num][snr_num].end_time)
      correct_this_trial = score_word_list(true_words, recognized_words,
                                           max_count=5)
      correct_count += correct_this_trial
      if debug:
        print(f'SNR {snr}:')
        print(f'  Expected words: {true_words}')
        print(f'  Recognized words: {recognized_words}')
        print(f'  Correct count is {correct_this_trial}')
    correct_counts.append(correct_count)
  correct_frac = np.asarray(correct_counts,
                            dtype=float) / (num_keywords*num_lists)
  return correct_frac


# Models listed here:
# https://cloud.google.com/speech-to-text/docs/transcription-model

all_model_names = ('latest_long',
                   'latest_short',
                   # 'command_and_search',
                   # 'phone_call',
                   'telephony',
                   # 'video',
                   'medical_dictation',
                   'medical_conversation',
                   # 'default',
                   'chirp',
)

def recognize_with_all_models(
    project_id: str,
    spin_test_names: List[str],
    model_names: List[str] = all_model_names) -> Dict[str, SpinFileTranscripts]:
  """Recognize all QuickSIN test files (from the spin_test_names argument)
  Return a dictionary of scores vs. list of lists of transcripts, keyed by 
  the model name.
  """
  model_results = {}

  for model_name in model_names:
    print(f'Model: {model_name}')
    asr_engine = RecognitionEngine()
    asr_engine.CreateSpeechClient(project_id, model=model_name)
    asr_engine.CreateRecognizer()
    babble_transcripts = recognize_all_spin(spin_test_names, asr_engine)
    # Babble_transcripts is a SpinFileTranscripts List[List[RecogResults]]
    model_results[model_name] = babble_transcripts
  return model_results


def score_all_models(
    model_results: Dict[str, SpinFileTranscripts],
    ground_truths: List[List[SpinSentence]],
    test_snrs: List[float] = spin_snrs) -> Dict[str, np.ndarray]:
  """Score all QuickSIN test files (from the spin_test_names argument) against
  the ground_truths.  Return a dictionary of scores vs. SNRs, keyed by the 
  model name.
  """
  model_scores = {}
  print(type(model_results), model_results)

  for model_name in model_results:
    babble_transcripts = model_results[model_name]
    scores = score_all_tests(test_snrs,
                             ground_truths,
                             babble_transcripts,
                             debug=True)
    model_scores[model_name] = scores
  return model_scores


def save_recognition_results(
    recognition_results: Dict[str, SpinFileTranscripts],
    recognition_json_file: str):
  """Save the recognition results for all models into a JSON file so we don't 
  have to query the cloud again."""
  class DataclassEncoder(json.JSONEncoder):
    def default(self, o):
      if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
      elif isinstance(o, set):
        return list(o)
      return super().default(o)

  saved_data = {
    'recognition_results': recognition_results,
    'time': str(datetime.datetime.now())
  }
  with fsspec.open(recognition_json_file, 'w') as fp:
    json.dump(saved_data, fp, cls=DataclassEncoder)

def load_recognition_results(filename: str) -> Dict[str, SpinFileTranscripts]:
  """Load the precomputed QuickSIN results from a file."""
  with fsspec.open(filename, 'r') as fp:
    all_results = json.load(fp)
  if 'recognition_results' in all_results:
    results = all_results['recognition_results']
  print(f'Reloading recognition results saved at {all_results["time"]}')
  for k in results:
    # print(type(results[k]), results[k])
    list_of_lists = []
    for i in results[k]:
      list_of_words = []
      for j in i:
        list_of_words.append(RecogResult(**j))
      list_of_lists.append(list_of_words)
    results[k] = list_of_lists
  return results

def save_model_scores(scores: Dict[str, np.ndarray], filename: str):
  """Save the QuickSIN results for all models into a JSON file so we don't have
  to compute it again."""
  class NumpyArrayEncoder(json.JSONEncoder):
    def default(self, o):
      if isinstance(o, np.ndarray):
        return o.tolist()
      return json.JSONEncoder.default(self, o)

  saved_data = {
    'model_results': scores,
    'time': str(datetime.datetime.now())
  }
  with fsspec.open(filename, 'w') as fp:
    json.dump(saved_data, fp, cls=NumpyArrayEncoder)


def load_model_scores(filename: str) -> Dict[str, np.ndarray]:
  """Load the precomputed QuickSIN results from a file."""
  with fsspec.open(filename, 'r') as fp:
    all_results = json.load(fp)
  if 'model_results' in all_results:
    results = all_results['model_results']
  print(f'Reloading model scores saved at {all_results["time"]}')
  for k in results:
    results[k] = np.asarray(results[k])
  return results


###### LOGISTIC Functions ####
# Code and explanation from
# https://www.linkedin.com/pulse/how-fit-your-data-logistic-function-python-carlos-melus/


def logistic_curve(x: np.ndarray,
                   a: float, b: float, c:float, d: float) -> float:
  """
  Logistic function with parameters a, b, c, d
  a is the curve's maximum value (top asymptote)
  b is the curve's minimum value (bottom asymptote)
  c is the logistic growth rate or steepness of the curve
  d is the x value of the sigmoid's midpoint
  """
  return ((a-b) / (1 + np.exp(-c * (x - d)))) + b

def psychometric_curve(x, c, d):
  """Like the logistic curve above, but the output is always >= 0.0 and <= 1.0.
  """
  return logistic_curve(x, 1, 0, c, d)


def compute_quicksin_regression(snrs: List[float],
                                scores: np.ndarray) -> float:
  """Use regression to fit a logistic curve to the raw scores (vs. SNR)
  Return the SNR that gives 50% error.
  """
  #pylint: disable=unbalanced-tuple-unpacking
  logistic_params, _ = curve_fit(psychometric_curve,
                                 snrs,
                                 scores,
                                 ftol=1e-4)
  _, d = logistic_params
  return d

def run_ground_truth(ground_truth_json_file: str,
                     project_id: str,
                     sin_wav_dir: str = ('/content/drive/MyDrive/'
                                         'Stanford/QuickSIN22/'),
                     sentence_boundary_graph: str = '',
                    ):
  if not os.path.exists(ground_truth_json_file):
    truths = compute_quicksin_truth(
      sin_wav_dir,
      project_id,
      sentence_boundary_graph=sentence_boundary_graph)
    assert isinstance(truths, list)
    save_ground_truth(truths, ground_truth_json_file)
  else:
    truths = load_ground_truth(ground_truth_json_file)
    assert isinstance(truths, list)
    assert len(truths), 12
  return truths

def run_recognize_models(recognition_json_file: str,
                         audio_dir: str,
                         project_id: str) -> Dict[str, SpinFileTranscripts]:
  if not os.path.exists(recognition_json_file):
    spin_pattern = os.path.join(audio_dir, '*.wav')
    spin_file_names = fsspec.open_files(spin_pattern)
    spin_file_names = [f.full_name for f in spin_file_names]
    spin_test_names = [f for f in spin_file_names if 'Babble' in f]
    spin_test_names.sort(key=sort_by_list_number)

    recognition_results = recognize_with_all_models(
      project_id,
      spin_test_names,
      )
    save_recognition_results(recognition_results, recognition_json_file)
  else:
    recognition_results = load_recognition_results(recognition_json_file)
    assert isinstance(recognition_results, dict)
  return recognition_results

def run_score_models(models_json_file: str,
                     model_results: Dict[str, SpinFileTranscripts],
                     truths) -> Dict[str, np.ndarray]:
  if not os.path.exists(models_json_file):
    model_frac_scores = score_all_models(model_results,
                                         truths)
    print('Model fraction correct scores:', model_frac_scores)
    save_model_scores(model_frac_scores, models_json_file)
  else:
    model_frac_scores = load_model_scores(models_json_file)
    assert isinstance(model_frac_scores, dict)
  return model_frac_scores


# From: https://devarea.com/linear-regression-with-numpy/
def linear_regression(x: np.ndarray, y: np.ndarray) -> Tuple[float, float]:
  x = np.asarray(list(x))
  y = np.asarray(list(y))
  # https://stackoverflow.com/questions/44462766/removing-nan-elements-from-2-arrays
  indices = np.logical_not(np.logical_or(np.isnan(x), np.isnan(y)))    
  indices = np.array(indices)
  x = x[indices]
  y = y[indices]
  m = (len(x) * np.sum(x*y) - np.sum(x) * np.sum(y)) / (len(x)*np.sum(x*x) - np.sum(x) ** 2)
  b = (np.sum(y) - m *np.sum(x)) / len(x)
  return m, b

#############  MAIN PROGRAM - Test all models and create graphs  ###############

FLAGS = flags.FLAGS

# Flag names are globally defined!  So in general, we need to be
# careful to pick names that are unlikely to be used by other libraries.
# If there is a conflict, we'll get an error at import time.
flags.DEFINE_string('ground_truth_cache', 'ground_truth.json',
                    'Where to keep a cache of the QuickSIN ground truth')
flags.DEFINE_string('model_recognition_cache', 'model_recognition.json',
                    'Where to keep a cache of the QuickSIN model results')
flags.DEFINE_string('model_result_cache', 'model_result.json',
                    'Where to keep a cache of the QuickSIN model results')
flags.DEFINE_string('audio_dir', '../QuickSIN/QuickSIN22/',
                    'Where to find the QuickSIN .wav files')
flags.DEFINE_string('sentence_boundary_graph',
                    'results/sentence_boundaries.png',
                    'Where to store the sentence boundary debugging graph')
flags.DEFINE_string('all_score_graph', 'results/all_score_graph.png',
                    'Where to store the plot with all the scores.')
flags.DEFINE_string('spin_logistic_graph', 'results/spin_logistic_graph.png',
                    'Where to store the plot with logistic SPIN scores.')
flags.DEFINE_string('spin_counting_graph', 'results/spin_counting_graph.png',
                    'Where to store the plot with logistic SPIN scores.')
flags.DEFINE_string('logistic_counting_graph',
                    'results/logistic-counting-comparison.png',
                    'Graph comparing regression vs. counting results')
flags.DEFINE_string('logistic_fit_graph',
                    'results/logistic_fit.png',
                    'Graph showing logistic regression fit to QuickSIN data')
flags.DEFINE_boolean('debug', False, 'Produces debugging output.')
flags.DEFINE_float('human_level', 2.0,
                   'Normal human performance so we can subtract it (dB)')


def main(_):
  project_id = os.getenv('GOOGLE_CLOUD_PROJECT')
  assert project_id, 'Can not find GOOGLE_CLOUD_PROJECT.'

  truths = run_ground_truth(FLAGS.ground_truth_cache,
                            project_id,
                            FLAGS.audio_dir,
                            FLAGS.sentence_boundary_graph)

  model_results = run_recognize_models(FLAGS.model_recognition_cache,
                                       FLAGS.audio_dir,
                                       project_id)

  model_frac_scores = run_score_models(FLAGS.model_result_cache,
                                       model_results,
                                       truths)

  #pylint: disable=consider-using-dict-items
  if FLAGS.all_score_graph:
    plt.clf()
    for m in model_frac_scores:
      plt.plot(spin_snrs,
               model_frac_scores[m],
               label=m)
    plt.xlabel('SNR (dB)')
    plt.ylabel('Fraction of words recognized correctly')
    plt.legend()
    plt.savefig(FLAGS.all_score_graph)

  quicksin_regression_loss = {}
  for m in model_frac_scores:
    quicksin_regression_loss[m] = compute_quicksin_regression(
      spin_snrs, model_frac_scores[m]) - FLAGS.human_level

  if FLAGS.spin_logistic_graph:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    bar_labels = [s.replace('_', '\n') for s in quicksin_regression_loss]
    bar_container = ax.bar(bar_labels, quicksin_regression_loss.values())
    ax.set(ylabel='QuickSIN Loss (dB)',
           title='Cloud ASR QuickSIN Scores (logistic)', ylim=(0, 16))
    ax.bar_label(bar_container)
    a = plt.axis()
    plt.plot(a[:2], [15, 15], '--', label='Severe SNRloss')
    plt.plot(a[:2], [7, 7], '--', label='Moderate SNRloss')
    plt.plot(a[:2], [3, 3], '--', label='Mild SNRloss')
    plt.legend()
    plt.axis(a)
    plt.savefig(FLAGS.spin_logistic_graph)

  if FLAGS.logistic_fit_graph:
    scores = model_frac_scores['latest_long']
    # pylint: disable=unbalanced-tuple-unpacking
    logistic_params, _ = curve_fit(psychometric_curve,
                                   spin_snrs,
                                   scores,
                                   ftol=1e-4)
    detailed_snr = np.arange(0, 25, 0.1)
    fig = plt.figure(figsize=(6.4, 4.8))  # Reset to default size
    plt.plot(spin_snrs, scores, 'x', label='Experimental Data')
    plt.plot(detailed_snr,
             psychometric_curve(detailed_snr,
                                logistic_params[0],
                                logistic_params[1]),
             label='Logistic Fit')
    plt.plot([0, 25], [0.5, 0.5], '--', label='50% Theshold')
    plt.plot([logistic_params[1], logistic_params[1]], [0, 0.5], ':')
    plt.legend()
    plt.xlabel('SNR (dB)')
    plt.ylabel('Fraction recognized correctly')
    plt.title('Logistic Regression for QuickSIN Data')
    plt.savefig(FLAGS.logistic_fit_graph)

  quicksin_counting_loss = {}
  for m in model_frac_scores:
    # if m == 'latest_short': continue
    # Translate fraction correct into the average number of correct
    # words per SNR across all lists.
    scores = model_frac_scores[m]
    assert len(scores) == 6, f'Not enough scores {scores} for model {m}'
    snr50 = 27.5 - 5 * np.sum(model_frac_scores[m])
    quicksin_counting_loss[m] = snr50 - FLAGS.human_level

  if FLAGS.spin_counting_graph:
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    bar_labels = [s.replace('_', '\n') for s in quicksin_counting_loss]
    bar_container = ax.bar(
      bar_labels,
      quicksin_counting_loss.values())
    ax.set(ylabel='QuickSIN Loss (dB)',
           title='Cloud ASR QuickSIN Scores (counting)', ylim=(0, 16))
    ax.bar_label(bar_container)
    a = plt.axis()
    plt.plot(a[:2], [15, 15], '--', label='Severe SNRloss')
    plt.plot(a[:2], [7, 7], '--', label='Moderate SNRloss')
    plt.plot(a[:2], [3, 3], '--', label='Mild SNRloss')
    plt.legend()
    plt.axis(a)
    plt.savefig(FLAGS.spin_counting_graph)

  if FLAGS.logistic_counting_graph:
    plt.clf()
    # Remove outlier from this comparison
    quicksin_counting_loss['latest_short'] = np.nan
    plt.plot(quicksin_regression_loss.values(),
             quicksin_counting_loss.values(), 'x')
    plt.xlabel('QuickSIN loss by logistic regression (dB)')
    plt.ylabel('QuickSIN loss by counting (dB)')
    plt.title('Comparison of loss by counting and logistic regression')
    plt.axis('square')
    current_axis = plt.axis()
    left = max(current_axis[0], current_axis[2])
    right = min(current_axis[1], current_axis[3])
    plt.plot([left, right], [left, right], '--')
    plt.savefig(FLAGS.logistic_counting_graph)

    m, b = linear_regression(quicksin_regression_loss.values(),
                             quicksin_counting_loss.values())
    print('Linear regression connecting logistic and counting approaches: '
          f'slope is {m}, bias is {b}')

if __name__ == '__main__':
  app.run(main)

