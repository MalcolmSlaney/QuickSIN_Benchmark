"""Code to test the auditory toolbox."""
import os

from absl.testing import absltest
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
      recognizer=f"projects/{project_id}/locations/global/recognizers/_",
      config=config,
      content=content,
  )

  # Transcribes the audio into text
  response = client.recognize(request=request)

  for result in response.results:
      print(f"Transcript: {result.alternatives[0].transcript}")

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
  with open(audio_file, "rb") as f:
      content = f.read()

  config = cloud_speech.RecognitionConfig(
      auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
      language_codes=["en-US"],
      model="chirp",
  )

  request = cloud_speech.RecognizeRequest(
      recognizer=f"projects/{project_id}/locations/us-central1/recognizers/_",
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
    self.assertEqual(alternative.transcript, 'a huge tapestry hung in her hallway')
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

  def test_spin_labels(self):
    self.assertEqual(gasr.word_alternatives('foo'), ['foo',])
    self.assertEqual(gasr.word_alternatives('foo/bar'), ['foo', 'bar'])

    self.assertEqual(gasr.all_keyword_dict[1, 0],
                    [['tear', 'tara'], ['thin'], ['sheet'], ['yellow'], ['pad']])

if __name__ == '__main__':
  absltest.main()
