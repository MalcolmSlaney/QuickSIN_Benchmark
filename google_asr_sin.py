"""Test Google Cloud ASR offerings on the Speech in Noise (SPIN) test."""

import json
import re
from typing import List, Dict, Optional, Tuple, Union

import dataclasses
from scipy import signal
from scipy.io import wavfile
# from scipy.optimize import curve_fit
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

class RecognitionEngine(object):
  """A class that provides a nicer interface to Google's Cloud
  text-to-speech API.
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
    except:
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


####### Organize SPIN recogntion results ########

# A list of lists.  Each (final) list is a list of recognition results (words
# and times). Then a list of these "sentence" lists.
SpinFileTranscripts = List[List[RecogResult]]

def recognize_all_spin(all_wavs: List[str],
                       asr_engine: RecognitionEngine,
                       debug=False) -> SpinFileTranscripts:
  """Recognize some SPiN sentences using the specified ASR engine.
  Return the transcription results in a dictionary keyed by the
  (last part of) the filename."""
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
    spin_truth_names: List[str]) -> Tuple[list[int], np.ndarray]:
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
      _, audio_data = wavfile.read(fp)
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
    # plt.plot(y[start_sample:end_sample])
    i = np.argmin(y[start_sample:end_sample]) + start_sample
    return i

  # Look for the minimum in each approximate range.
  splits = np.array([0.2, 0.3, 0.5, 0.7, 0.9, 1.1])*1e6
  breaks = [0]
  for i in range(5):
    breaks.append(find_min(envelope, splits[i], splits[i+1]))
  breaks.append(max_len)

  return breaks, all_audio

#################### SPIN TESTS ############################

# Pages 111 and 112 of this PDF:
# https://etda.libraries.psu.edu/files/final_submissions/5788

key_word_list = """
L 0 S 0  white silk jacket any shoes
L 0 S 1  child crawled into dense grass
L 0 S 2  Footprints show/showed path took beach
L 0 S 3  event near edge fresh air
L 0 S 4  band Steel 3/three inches/in wide
L 0 S 5  weight package seen high scale

L 1 S 0  tear/Tara thin sheet yellow pad
L 1 S 1  cruise Waters Sleek yacht fun
L 1 S 2  streak color down left Edge
L 1 S 3  done before boy/boys see it
L 1 S 4  Crouch before jump miss mark
L 1 S 5  square peg settle round hole

L 2 S 0  pitch straw through door stable
L 2 S 1  sink thing which pile/piled dishes
L 2 S 2  post no bills office wall
L 2 S 3  dimes showered/shower down all sides
L 2 S 4  pick card slip under pack/Pact
L 2 S 5  store jammed before sale start

L 3 S 0  sense smell better than touch
L 3 S 1  picked up dice second roll
L 3 S 2  drop ashes worn/Warren Old rug
L 3 S 3  couch cover Hall drapes blue
L 3 S 4  stems Tall Glasses cracked broke
L 3 S 5  cleats sank/sink deeply soft turf

L 4 S 0  have better than wait Hope
L 4 S 1  screen before fire kept Sparks
L 4 S 2  thick glasses helped/help read print/prints
L 4 S 3  chair looked strong no bottom
L 4 S 4  told wild Tales/tails frighten him
L 4 S 5  force equal would move Earth

L 5 S 0  leaf drifts along slow spin
L 5 S 1  pencil cut sharp both ends
L 5 S 2  down road way grain farmer
L 5 S 3  best method fix place clips
L 5 S 4  if Mumble your speech lost
L 5 S 5  toad Frog hard tell apart

L 6 S 0  kite dipped swayed/suede stayed aloft/loft
L 6 S 1  beatle/beetle drowned hot June/Tunes sun/son
L 6 S 2  theft Pearl pin Kept Secret
L 6 S 3  wide grin earned many friends
L 6 S 4  hurdle pit aid long Pole
L 6 S 5  Peep/keep under tent see Clown

L 7 S 0  sun came light Eastern sky
L 7 S 1  stale smell old beer lingers
L 7 S 2  desk firm on shaky floor
L 7 S 3  list names carved around base
L 7 S 4  news struct/struck out Restless Minds
L 7 S 5  Sand drifts/Drift over sill/sale house

L 8 S 0  take shelter tent keep still
L 8 S 1  Little Tales/tails they tell false
L 8 S 2  press pedal with left foot
L 8 S 3  black trunk fell from Landing/landings
L 8 S 4  cheap clothes flashy/flash don't last
L 8 S 5  night alarm roused/roust deep sleep

L 9 S 0  dots light betray/betrayed black cat
L 9 S 1  put chart mantle Tack down
L 9 S 2  steady drip worse drenching rain
L 9 S 3  flat pack less luggage space
L 9 S 4  gloss/glass top made unfit read
L 9 S 5  Seven Seals stamped great sheets

L10 S 0  marsh freeze when cold enough
L10 S 1  gray mare walked before colt/cold
L10 S 2  bottles hold four/for kinds rum
L10 S 3  wheeled/wheled bike past winding road
L10 S 4  throw used paper cup plate
L10 S 5  wall phone ring loud often

L11 S 0  hinge door creaked old age
L11 S 1  bright lanterns Gay dark lawn
L11 S 2  offered proof  form large chart
L11 S 3  their eyelids droop/drop want sleep
L11 S 4  many ways do these things
L11 S 5  we like see clear weather
""".split('\n')

def word_alternatives(words) -> List[str]:
  """Convert a string with words separated by '/' into a tuple."""
  if '/' in words:
    return words.split('/')
  return [words,]


def ingest_spin_keyword_lists(word_list: str) -> Dict[Tuple[int, int],
                                                      List[List[str]]]:
  """Convert the text from the big string above into a list of key words 
  (and alternatives) that describe the expected answers from a SPIN test.
  """
  keyword_dict = {}
  for line in word_list:
    line = line.strip().lower()
    if not line: continue
    list_number = int(line[1:3])
    sentence_number = int(line[5:7])
    key_words = line[7:].split(' ')
    key_words = [w for w in key_words if w]
    key_list = [word_alternatives(w) for w in key_words]
    if len(key_list) != 5:
      print(f'Have too many words in L{list_number} S{sentence_number}:',
            key_list)
    keyword_dict[list_number, sentence_number] = key_list
  return keyword_dict

all_keyword_dict = ingest_spin_keyword_lists(key_word_list)

######## Recognize the SPIN waveforms and calculate all word timings ##########

@dataclasses.dataclass
class SpinSentence:
  """A structure that describes one SPiN sentence, with the transcript,
  individual words, the sentence start and end time, and the SNR.

  There are six SPiN sentences per list, one per SNR.
  """
  sentence_words: List[str]
  true_word_list: List[List[str]]  # List of words and their alternatives
  # words: list[str]
  start_time: float
  end_time: float
  snr: float  # This sentence's test SNR


# Organize the clean speech transcripts.  Each 60s wavedform becomes a list of
# recognized sentences.  Return a list of list of sentences.

spin_snrs = (25, 20, 15, 10, 5, 0)

def format_quicksin_truth(
    spin_transcripts: SpinFileTranscripts,  # List of List of RecogResults
    sentence_breaks: List[float],  # Times in seconds
    snr_list: Tuple[float] = spin_snrs) -> List[List[SpinSentence]]:
  """Parse the recognition results and produce a List (of sentences at different
  SNRs).  Return a list of 12 SPIN lists, each list containing the 6 SPIN 
  sentences at the different SNRs.
  """
  assert len(spin_transcripts) > 0
  # assert len(sentence_breaks) == 7  # Not 7 for testing
  # assert len(snr_list) == 6         # Not 6 for testing
  spin_results = []
  # Iterate through the lists (each list contains 6 different sentences)
  for list_number, clean_transcript in enumerate(spin_transcripts):
    sentences = []
    for snr_number, snr in enumerate(snr_list):
      sentence_start_time = sentence_breaks[snr_number]
      sentence_end_time = sentence_breaks[snr_number+1]
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
  class EnhancedJSONEncoder(json.JSONEncoder):
    def default(self, o):
      if dataclasses.is_dataclass(o):
        return dataclasses.asdict(o)
      return super().default(o)

  with fsspec.open(filename, 'w') as fp:
    json.dump(truth, fp, cls=EnhancedJSONEncoder)


def load_ground_truth(filename: str) -> List[List[SpinSentence]]:
  """Load the precomputed QuickSIN ground truth from a file."""
  with fsspec.open(filename, 'r') as fp:
    truth = json.load(fp)
  for i in range(len(truth)):        # Nominally 12, except during testing
    for s in range(len(truth[i])):   # Nominally 6, except during testing
      truth[i][s] = SpinSentence(**truth[i][s])
  return truth


def compute_quicksin_truth(
    wav_dir: str,
    project_id: str,
    sentence_breaks: Optional[List[float]] = None,
    snr_list: Tuple[float] = spin_snrs) -> List[List[SpinSentence]]:
  """Create the ground truth for a SPIN test. 
  Process all the clean speech files to figure out the start and stop of each
  sentence.  Combine with the keyword list to create a list (by QuickSin list) 
  of lists of sentences (one sentence per test SNR).
  """
  spin_file_names = fsspec.open_files(os.path.join(wav_dir, '*.wav'))
  spin_file_names = [f.full_name for f in spin_file_names]
  spin_truth_names = [f for f in spin_file_names if 'Clean' in f]
  assert spin_truth_names, f'Could not find clean speech files in {wav_dir}.'
  print(f'Found {len(spin_truth_names)} QuickSIN lists to process.')

  if sentence_breaks is None:
    print('Finding sentence boundaries...')
    sentence_breaks, _ = find_sentence_boundaries(spin_truth_names)
    
    # TODO(malcolm): Perhaps this should be done in find_sentence_boundaries?!?
    sentence_breaks = [b/22050.0 for b in sentence_breaks]
    print('Sentence breaks are:', sentence_breaks)

  print('Transcribing the QuickSIN WAV files....')
  model = 'latest_long'
  asr_engine = RecognitionEngine()
  asr_engine.CreateSpeechClient(project_id, model)
  asr_engine.CreateRecognizer(with_timings=True)

  true_transcripts = recognize_all_spin(spin_truth_names, asr_engine)
  print(f'True transcripts are:', true_transcripts)

  print('Formatting the QuickSIN Ground Truth....')
  spin_ground_truths = format_quicksin_truth(true_transcripts,
                                             sentence_breaks,
                                             snr_list)
  return spin_ground_truths


def xx_create_ground_truth(
    # spin_truth_names: List[str],
    true_transcripts: SpinFileTranscripts) -> List[List[SpinSentence]]:
  ground_truths = []
  for list_number in range(12):
    filename = f'Clean List {list_number+1}.wav'
    sentences = []
    skip = 0  # Hack.. need to skip some results if empty.
    for snr_number, snr in enumerate(spin_snrs):
      print(f'Processing {filename} with SNR #{snr_number}')
      snr = spin_snrs[snr_number]
      alternatives = true_transcripts[filename].results[snr_number].alternatives
      if not alternatives:
        print(f'  Skipping an empty result for list {list_number} '
              f'snr {snr}')
        skip += 1
      elif alternatives[0].transcript == 'bring it to':
        print(f'  Skipping an extraneous result for list {list_number} '
              f'snr {snr}')
        skip += 1
      elif list_number == 4 and alternatives[0].transcript == 'finish':
        # Not sure why there is a whole bunch of reco before the
        # words start in Clean 5.
        skip += 1
      one_result = alternatives[0]
      transcript = one_result.transcript
      # all_words = [r.word.lower() for r in one_result.words]
      start_times = [parse_time(r.start_offset) for r in one_result.words]
      end_times = [parse_time(r.end_offset) for r in one_result.words]

      # if list_number == 1 and snr_number == 0:
      #   print(one_result)
      sentences.append(SpinSentence(transcript,
                                    all_keyword_dict[list_number, snr_number],
                                    # all_words,
                                    min(start_times), max(end_times), snr))
    assert len(sentences) == len(spin_snrs)
    ground_truths.append(sentences)
  return ground_truths

def xx_parse_spin_results(all_transcripts) -> List[List[RecogResult]]:
  """Using the recognized babble results, Create a list of lists.  One list for
  each of the 12 60s SPiN lists.  For each SPiN list, create a separate list for
  each SNR."""
  reco_results = []
  for i in range(12):
    filename = f'Babble List {i+1}.wav'
    # print(filename)
    test_results = all_transcripts[filename]
    test_transcript = parse_transcript(test_results)
    reco_results.append(test_transcript)
  return reco_results


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


def score_word_list(true_words: List[List[str]],
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
  for words_and_alternates in true_words:
    for word in words_and_alternates:
      if word in recognized_words:
        score += 1
        break
  if max_count:
    score = min(score, max_count)
  return score

def score_all_tests(snrs: List[float], ground_truths, reco_results,
                    debug=False):
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
