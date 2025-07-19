import pandas as pd

def analyze_campaigns(file_path='data/email_results.csv'):
    df = pd.read_csv(file_path)
    print("\nEmail Campaign Analyzer\n")
    print(f"Total campaigns: {df['campaign_id'].nunique()}")
    avg_open = df['open_rate'].mean()
    avg_click = df['click_rate'].mean()
    print(f"Avg Open rate: {avg_open:.2f}%")
    print(f"Avg Click rate: {avg_click:.2f}%")

    top_campaign = df.sort_values('click_rate', ascending=False).iloc[0]
    print(f"\nTop Campaign: {top_campaign['subject']} (Click rate: {top_campaign['click_rate']}%)")

    # NLP: Find most effective subject phrases
    from collections import Counter
    words = []
    for subject in df['subject']:
        words += subject.lower().split()
    top_words = Counter(words).most_common(5)
    print("\nTop subject line keywords:")
    for word, count in top_words:
        print(f"- {word}: {count}")

def recommend_time(df):
    best_time = df.groupby('send_hour')['open_rate'].mean().idxmax()
    print(f"\nRecommended send hour: {best_time}:00")

def main():
    file_path = 'data/email_results.csv'
    df = pd.read_csv(file_path)
    analyze_campaigns(file_path)
    recommend_time(df)

if __name__ == '__main__':
    main()
