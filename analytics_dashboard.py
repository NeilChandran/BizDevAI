import pandas as pd

def pipeline_conversion_rate(df):
    print("Calculating pipeline conversion rates...\n")
    stages = ['contacted', 'qualified', 'proposal', 'won']
    for i in range(1, len(stages)):
        prev_stage = stages[i-1]
        curr_stage = stages[i]
        prev = df[df['stage'] == prev_stage].shape[0]
        curr = df[df['stage'] == curr_stage].shape[0]
        if prev > 0:
            rate = 100 * curr / prev
        else:
            rate = 0
        print(f"{prev_stage} → {curr_stage}: {rate:.1f}%")

def top_industries(df, n=5):
    res = df['industry'].value_counts().head(n)
    print("\nTop Industries:\n", res)

def avg_deal_size(df):
    print(f"\nAverage deal size: ${df['deal_size'].mean():.2f}")

def dashboard():
    df = pd.read_csv('data/all_leads_cleaned.csv')
    print("BIZDEV AI INSIGHT DASHBOARD\n")
    print(f"Total leads: {df.shape[0]}")
    print(f"Wins: {df[df['deal_won'] == 1].shape[0]}\n")

    pipeline_conversion_rate(df)
    top_industries(df)
    avg_deal_size(df)

if __name__ == '__main__':
    dashboard()

