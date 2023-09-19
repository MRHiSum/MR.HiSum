## Mr.HiSum Dataset Details
------
### Metadata of Mr.HiSum

 `metadata.csv` looks like below

| video_id | yt8m_file | random_id | youtube_id   | duration | views  | labels         |
|----------|-----------|-----------|--------------|----------|--------|----------------|
| video_1  | train0026 | ORaA      | JhdjUam0l6A  | 258      | 84554  | [8]            |
| video_2  | train0026 | IwaA      | wq7rSbQx2G8  | 137      | 170768 | [11...]        |
| ...      | ...       | ...       | ...          | ...      | ...    | ...            |
| video_31892  | train3017 | ShWP | ynmR_tomXP8 | 142 | 59651 | [5..] |

- `video_id` is a unque index of the video used in this dataset.
- `yt8m_file` is a YouTube-8M tfrecord file where the video belongs to.
- `random_id` is a unique id used in the YouTube-8M dataset.
- `youtube_id` is a unique video id used in [youtube.com](youtube.com).
- `duration` is a length of the video in seconds.
- `views` is the view counts of the video. Every video has at least 50,000 views.
- `labels` is a list of class labels from YouTube-8M. You can check the name of each label from [YouTube-8M](https://research.google.com/youtube8m/) webpage 'Dataset Vocabulary' part.

----

### Most replayed crawler

Utilizing the YouTube video id, you can crawl 'Most replayed' statistics using our crawler.

```
python dataset/crawler.py --vid <video_id>
```
For example,
```
python dataset/crawler.py --vid JhdjUam0l6A
```
----
