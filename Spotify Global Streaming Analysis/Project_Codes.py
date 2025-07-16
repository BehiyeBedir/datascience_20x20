import matplotlib.pyplot as plt
import pandas as pd

# total streams by artist
df = pd.read_csv('/kaggle/input/spotify-charts/charts.csv')

artist_streams = df.groupby('artist')['streams'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
artist_streams.plot(kind='bar', color='purple')
plt.title('Top 10 artists with the most streams')
plt.ylabel('Total number of streams')
plt.xlabel('Artist')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()




# Calculate counts of MOVE_UP and MOVE_DOWN in the 'trend' column grouped by artist
trend_counts = df.groupby(['artist', 'trend']).size().unstack(fill_value=0)

# Make sure columns for MOVE_UP and MOVE_DOWN exist (some artists may not have them)
for col in ['MOVE_UP', 'MOVE_DOWN']:
    if col not in trend_counts.columns:
        trend_counts[col] = 0

# Select top 10 artists with the highest MOVE_UP counts
top_artist_move_up = trend_counts.sort_values(by='MOVE_UP', ascending=False).head(10)

# Plotting
plt.figure(figsize=(12,6))
top_artist_move_up[['MOVE_UP', 'MOVE_DOWN']].plot(kind='bar')
plt.title('Top 10 Fastest-Rising Artists Based on MOVE_UP Counts')
plt.ylabel('Count of Position Changes')
plt.xlabel('Artist')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Combine song title and artist for more clarity
df['song_artist'] = df['title'] + ' - ' + df['artist']

# Top 10 most streamed songs by total stream count
top_songs = df.groupby('song_artist')['streams'].sum().sort_values(ascending=False).head(10)

plt.figure(figsize=(12,6))
top_songs.plot(kind='bar', color='teal')
plt.title('Top 10 Most Streamed Songs (with Artists)')
plt.ylabel('Total Stream Count')
plt.xlabel('Song - Artist')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()






