"""
This file is only used to generate training data for a neural network, and thus
most not be used to download any content from https://tekstovi.net/2,0,0.html for
offline-browsing, as described in their license.
"""

import os
import threading
import requests
from typing import Optional, List, Callable, NoReturn
from bs4 import BeautifulSoup, element


class SongDataset(object):
    """
    Prepares the data set.
    """

    def __init__(self, output_file_path: str,
                 output_dir_path: str,
                 num_singers: int = 10000,
                 thread_count: int = 4,
                 save_separate_files: bool = True,
                 preprocessing_ops: Optional[List[Callable[[str], str]]] = None):
        """
        Constructor.
        :param output_file_path: Path to the file in which to store all the songs.
        :param output_dir_path: Path to the directory where to store songs grouped by the singer.
        :param num_singers: Number of singers whose songs to take.
        :param thread_count: Number of threads to use.
        :param save_separate_files: Whether to save each singer's songs in a separate file as well as the main file.
        :param preprocessing_ops: Preprocessing operations to apply on each song.
        """
        self._output_file_path = output_file_path
        self._output_dir_path = output_dir_path
        self._thread_count = thread_count
        self._lock = threading.Lock()
        self._terminal_mutex = threading.Lock()
        self._progress_mutex = threading.Lock()
        self._open_file = None
        self._save_separate = save_separate_files
        self._num_singers = num_singers
        self._preprocessing_ops = preprocessing_ops or []
        self._singers_processed = 0
        self._base_url = "https://tekstovi.net/"
        assert self._num_singers % self._thread_count == 0, "Number of singers must be divisible by the number of threads."

    def _get_url_with_singer(self, singer_id: int) -> str:
        """
        Creates url with the given singer id.
        :param singer_id: Id of the singer.
        :return: Generated url.
        """
        return "https://tekstovi.net/2,{},0.html".format(str(singer_id))

    def _preprocess_song(self, song_lyrics: element.Tag) -> Optional[str]:
        """
        Preprocesses a single song.
        :param song_lyrics: BeautifulSoup paragraph which contains a song.
        :return: Preprocessed song as a string.
        """
        if not song_lyrics:
            return None
        # apply preprocessing
        song = song_lyrics.text.strip()
        for preprocess_op in self._preprocessing_ops:
            song = preprocess_op(song)
        return song

    def _get_single_song(self, url: str) -> Optional[str]:
        """
        Scrapes a single song from the web and preprocesses it.
        :param url: Url of the song.
        :return: Preprocessed text of the song as a string.
        """
        page = requests.get(url)
        soup = BeautifulSoup(page.content, 'html.parser')
        song_lyrics = soup.find(class_='lyric')
        return self._preprocess_song(song_lyrics)

    def _store_songs_for_singer(self, singer_id: int) -> NoReturn:
        """
        Stores all of the songs from a specified singer.
        :param singer_id: Id of the singer whose songs to save.
        :return: None.
        """
        # load page
        page = requests.get(self._get_url_with_singer(singer_id))
        soup = BeautifulSoup(page.content, 'html.parser')
        singer_name = soup.find(class_='lyricCapt')
        buffer = []
        num_songs = 0
        if singer_name:
            # for each of the singer's songs
            for song in soup.find_all(class_='artLyrList'):
                url = "{}{}".format(self._base_url, song.find('a')['href'])
                song = self._get_single_song(url)
                if song:
                    num_songs += 1
                    buffer.append(song)
            if buffer:
                # write output to singer file
                # no need to lock the mutex here because singer_file is thread local
                buffered_line = ''.join(buffer)
                if self._save_separate:
                    singer_output_file = os.path.join(self._output_dir_path,
                                                      f"{singer_name.text.strip()}_{singer_id}.txt")
                    with open(singer_output_file, 'w', encoding='utf-8') as singer_file:
                        singer_file.write(buffered_line)
                # lock the mutex and write song to a file
                with self._lock:
                    self._open_file.write(buffered_line)
        buffer.clear()
        with self._progress_mutex:
            self._singers_processed += 1
        if num_songs:
            with self._terminal_mutex:
                print(
                    f'== Downloaded {num_songs} songs from {singer_name.text.strip()} (processed {self._singers_processed}/{self._num_singers}). ==')

    def _scrape_range(self, start_index: int, end_index: int) -> NoReturn:
        """
        Scrapes all the songs from the singers whose ID is in the specified range.
        Used as a thread worker to split the load evenly between the threads.
        :param start_index: Start ID of the singer (inclusive).
        :param end_index: End ID of the singer (exclusive).
        :return: None.
        """
        for index in range(start_index, end_index):
            self._store_songs_for_singer(index)

    def prepare(self) -> NoReturn:
        """
        Prepares the raw data set and applies preprocessing.
        :return: None.
        """
        # create empty file to store the songs
        self._open_file = open(self._output_file_path, 'w', encoding='utf-8')
        # split the range evenly
        thread_split_count = self._num_singers // self._thread_count
        threads = []
        # create the threads
        for index in range(self._thread_count):
            t = threading.Thread(target=self._scrape_range,
                                 args=(index * thread_split_count, (index + 1) * thread_split_count))
            threads.append(t)
            t.start()
        # wait for the threads to finish
        for thread in threads:
            thread.join()
        self._open_file.close()
        print(f"Data set prepared. Output file generated: {self._output_file_path}")
