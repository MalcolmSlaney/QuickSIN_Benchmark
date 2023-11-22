"""Code to test the auditory toolbox."""
import io
import os
import re

from absl.testing import absltest
import contextlib
import fsspec
import numpy as np
from scipy.optimize import curve_fit

from typing import Optional

# import numpy as np
# import scipy

import google_asr_sin as gasr

from google.cloud.speech_v2 import SpeechClient
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions

GOOGLE_CLOUD_PROJECT = os.getenv('GOOGLE_CLOUD_PROJECT')
if not GOOGLE_CLOUD_PROJECT:
  raise ValueError('Can not find GOOGLE_CLOUD_PROJECT in shell environment.')

# From https://cloud.google.com/speech-to-text/docs/migration
def quickstart_v2(
    project_id: str,
    audio_file: str,
) -> cloud_speech.RecognizeResponse:
  """Transcribe an audio file."""
  # Instantiates a client
  client = SpeechClient()

  # Reads a file as bytes
  with open(audio_file, 'rb') as f:
    content = f.read()

  config = cloud_speech.RecognitionConfig(
      auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
      language_codes=['en-US'],
      model='long',
  )

  request = cloud_speech.RecognizeRequest(
      recognizer=f'projects/{project_id}/locations/global/recognizers/_',
      config=config,
      content=content,
  )

  # Transcribes the audio into text
  response = client.recognize(request=request)

  for result in response.results:
    print(f'Transcript: {result.alternatives[0].transcript}')

  return response


def transcribe_chirp(
    project_id: str,
    audio_file: str,
) -> cloud_speech.RecognizeResponse:
  """Transcribe an audio file using Chirp."""
  # Instantiates a client
  client = SpeechClient(
      client_options=ClientOptions(
          api_endpoint='us-central1-speech.googleapis.com',
      )
  )

  # Reads a file as bytes
  with open(audio_file, 'rb') as f:
    content = f.read()

  config = cloud_speech.RecognitionConfig(
      auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
      language_codes=['en-US'],
      model='chirp',
  )

  request = cloud_speech.RecognizeRequest(
      recognizer=f'projects/{project_id}/locations/us-central1/recognizers/_',
      config=config,
      content=content,
  )

  # Transcribes the audio into text
  response = client.recognize(request=request)

  for result in response.results:
    print(f'Transcript: {result.alternatives[0].transcript}')

  return response

class GoogleRecognizerTest(absltest.TestCase):
  """Test cases for auditory toolbox."""
  def test_google_generic_v2(self):
    """Test basic Google Cloud v2 speec-to-text functionality."""
    project_id = GOOGLE_CLOUD_PROJECT
    response = quickstart_v2(project_id, 'tests/tapestry.wav')
    self.assertLen(response.results, 1)
    self.assertEqual(response.results[0].alternatives[0].transcript,
                     'a huge tapestry hung in her hallway')

  def test_google_chirp(self):
    project_id = GOOGLE_CLOUD_PROJECT
    response = transcribe_chirp(project_id, 'tests/tapestry.wav')

    self.assertLen(response.results, 1)
    self.assertEqual(response.results[0].alternatives[0].transcript.strip(),
                     'a huge tapestry hung in her hallway')

  def test_asr_class(self):
    engine = gasr.RecognitionEngine()
    engine.CreateSpeechClient(GOOGLE_CLOUD_PROJECT, model='long')
    engine.CreateRecognizer()

    recognizers = engine.ListRecognizers(GOOGLE_CLOUD_PROJECT)
    self.assertLen(list(recognizers), 0)  # Not sure why we don't get any back.

    response = engine.RecognizeFile('tests/tapestry.wav',
                                    with_timings=True)
    # Make sure we get one result (sentence) and only one alternative result.
    self.assertLen(list(response.results), 1)
    self.assertLen(list(response.results[0].alternatives), 1)
    alternative = response.results[0].alternatives[0]

    # Make sure we get the correct transcription and the number of words
    self.assertEqual(alternative.transcript,
                     'a huge tapestry hung in her hallway')
    self.assertLen(alternative.words, 7)

    # Make sure the word start and end times make sense.
    start_times = [gasr.parse_time(w.start_offset) for w in alternative.words]
    end_times = [gasr.parse_time(w.end_offset) for w in alternative.words]
    self.assertTrue(end_times > start_times)
    self.assertGreaterEqual(start_times[0], 0.0)
    self.assertGreaterEqual(end_times[-1], 2.8)

  def test_parse_transcript(self):
    engine = gasr.RecognitionEngine()
    engine.CreateSpeechClient(GOOGLE_CLOUD_PROJECT, model='long')
    engine.CreateRecognizer()
    response = engine.RecognizeFile('tests/tapestry.wav',
                                    with_timings=True)

    self.assertIsInstance(response, cloud_speech.RecognizeResponse)

    asr_transcript = gasr.parse_transcript(response)
    # print(asr_transcript)
    # One extra word for the period.
    self.assertLen(asr_transcript, 8)

    self.assertEqual([w.word for w in asr_transcript],
                     'a huge tapestry hung in her hallway .'.split())
    start_times = [w.start_time for w in asr_transcript]
    end_times = [w.end_time for w in asr_transcript]
    self.assertTrue(end_times > start_times)
    self.assertGreaterEqual(start_times[0], 0.0)
    self.assertGreaterEqual(end_times[-1], 2.8)

  def test_spin_targets(self):
    self.assertEqual(gasr.word_alternatives('foo'), ['foo',])
    self.assertEqual(gasr.word_alternatives('foo/bar'), ['foo', 'bar'])

    self.assertEqual(gasr.all_keyword_dict[1, 0],
                     [['tear', 'tara'], ['thin'], ['sheet'],
                      ['yellow'], ['pad']])

  def test_ffmpeg_commands(self):
    # https://stackoverflow.com/questions/56045623/how-to-capture-the-stdout-stderr-of-a-unittest-in-a-variable
    f = io.StringIO()
    with contextlib.redirect_stdout(f):
      gasr.generate_ffmpeg_cmds()
    lines = f.getvalue().split('\n')

    clean_re = re.compile(r'Clean List (\d*).wav')
    babble_re = re.compile(r'Babble List (\d*).wav')
    def match_clean(line: str, match_re) -> Optional[int]:
      if line and len(line.strip()):
        m = match_re.search(line)
        if m:
          return int(m[1])
      return None

    # Make sure we produce all 12 "Clean List X.wav" files.
    matches = [match_clean(line, clean_re) for line in lines]
    matches = list(filter(lambda item: item is not None, matches))
    self.assertEqual(matches, list(range(1, 13)))

    # Make sure we produce all 12 "Babble List X.wav" files.
    matches = [match_clean(line, babble_re) for line in lines]
    matches = list(filter(lambda item: item is not None, matches))
    self.assertEqual(matches, list(range(1, 13)))

  def test_spin_files(self):
    score = gasr.score_word_list([['white', 'black'], ['silk'], ['jacket'],
                                  ['goes'], ['with'], ['any'], ['shoes']],
                                 ['a', 'black', 'silk', 'jacket'])
    self.assertEqual(score, 3)

    spin_file_names = fsspec.open_files('tests/*.wav')
    spin_file_names = [f.full_name for f in spin_file_names]

    # Make sure we find the right number of clean and babble test files.
    spin_test_names = [f for f in spin_file_names if 'Babble' in f]
    spin_truth_names = [f for f in spin_file_names if 'Clean' in f]
    self.assertLen(list(spin_test_names), 2)
    self.assertLen(list(spin_truth_names), 2)

    asr_engine = gasr.RecognitionEngine()
    asr_engine.CreateSpeechClient(GOOGLE_CLOUD_PROJECT, 'long')
    asr_engine.CreateRecognizer(with_timings=True)
    reco_transcripts = gasr.recognize_all_spin(spin_truth_names, asr_engine)

    # Make sure the SPIN files recognition results have the right form.
    self.assertIsInstance(reco_transcripts, list)
    self.assertLen(reco_transcripts, len(spin_truth_names))
    self.assertIsInstance(reco_transcripts[0], list)
    self.assertIsInstance(reco_transcripts[0][0], gasr.RecogResult)

    # Create the ground truth (correct words and their timing)
    spin_ground = gasr.create_spin_truth(reco_transcripts,
                                         [0, 236105/22050, 419853/22050],
                                         [20, 10])
    self.assertLen(spin_ground, 2)  # One for each input file

    self.assertLen(spin_ground[0], 2)  # Two sentences in partial files #1
    self.assertLen(spin_ground[1], 2)  # Two sentences in partial files #2

    self.assertIsInstance(spin_ground[0][0], gasr.SpinSentence)
    self.assertIsInstance(spin_ground[0][1], gasr.SpinSentence)
    self.assertIsInstance(spin_ground[1][0], gasr.SpinSentence)
    self.assertIsInstance(spin_ground[1][1], gasr.SpinSentence)

    # Make sure we get the right words from the transcdript
    found_words = gasr.words_in_trial(reco_transcripts[0], 5, 6, tolerance=0)
    print(f'Found words are: {found_words}')
    self.assertEqual(found_words, ['white', 'silk', 'jacket'])

    # Make sure we get more words with a bigger tolerance.
    found_words = gasr.words_in_trial(reco_transcripts[0], 5, 6, tolerance=1)
    print(f'Found words are: {found_words}')
    self.assertEqual(found_words, ['it', 'is', 'a', 'white', 'silk', 'jacket',
                                   'goes', 'with', 'any', 'shoes'])

    # Make sure we count the number of words right.
    true_words = spin_ground[0][0].true_word_list
    score = gasr.score_word_list(true_words, found_words)
    self.assertEqual(score, 5)

    # Now look at the wrong list.
    true_words = spin_ground[0][1].true_word_list
    score = gasr.score_word_list(true_words, found_words)
    self.assertEqual(score, 0)

    score = gasr.score_all_tests([20, 10], spin_ground, reco_transcripts,
                                 debug=True)


  def test_logistic_fit(self):
    t = np.arange(-10, 10, .01)

    a = 1  # fixed in this application
    b = 0  # Fixed in this application
    c = 3
    d = 4
    data = gasr.logistic_curve(t, a, b, c, d)

    # pylint: disable=unbalanced-tuple-unpacking
    logistic_params, _ = curve_fit(gasr.psychometric_curve,
                                            t,
                                            data,
                                            ftol=1e-4)
    est_c, est_d = logistic_params
    self.assertEqual(c, est_c)
    self.assertEqual(d, est_d)

if __name__ == '__main__':
  absltest.main()
