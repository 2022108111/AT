import jieba
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import random
from gensim import corpora, models
import re

# 加载文本和停用词
def load_file(file_path, encoding='utf-8'):
    with open(file_path, 'r', encoding=encoding) as file:
        return file.read()

def load_stopwords(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return set(line.strip() for line in file)

# 文本预处理，使用停用词
def preprocess_text(text, stopwords):
    text = re.sub(r'[^\w\s]', '', text)  # 删除非字词字符，保留文字和空格
    words = jieba.cut(text)
    return [word for word in words if word not in stopwords and len(word) > 1]

# 词频统计和可视化

# 设置matplotlib的字体
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为SimHei显示中文
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def plot_word_frequency(words):
    counter = Counter(words)
    labels, values = zip(*counter.most_common(20))
    
    plt.figure(figsize=(12, 8))
    plt.bar(labels, values, color='#f6bec8')  # 设置条形图颜色为指定的十六进制颜色代码
    plt.xlabel('词汇')
    plt.ylabel('频率')
    plt.title('词频统计')
    plt.xticks(rotation=45)
    plt.show()

# 情感分析和饼图可视化
def analyze_and_plot_sentiments(words, pos_words, neg_words):
    positive_count = sum(1 for word in words if word in pos_words)
    negative_count = sum(1 for word in words if word in neg_words)
    total = len(words)
    sentiment_counts = {
        'Positive': positive_count / total * 100,
        'Negative': negative_count / total * 100,
        'Neutral': 100 - (positive_count + negative_count) / total * 100
    }

    labels = sentiment_counts.keys()
    sizes = sentiment_counts.values()
    colors = ['#F9D3E3', '#C25160', '#F091A0']
    explode = (0.1, 0, 0)
    
    plt.figure(figsize=(7, 7))
    plt.pie(sizes, explode=explode, labels=labels, colors=colors, autopct='%1.1f%%', shadow=True, startangle=140)
    plt.axis('equal')
    plt.title('情感分布')
    plt.show()

# 情感分析
def analyze_sentiment(words, pos_words, neg_words):
    positive_count = sum(1 for word in words if word in pos_words)
    negative_count = sum(1 for word in words if word in neg_words)
    total = len(words)
    return {
        'Positive': positive_count / total * 100,
        'Negative': negative_count / total * 100,
        'Neutral': (total - positive_count - negative_count) / total * 100
    }

# 创建词云
def create_wordcloud(words, colors):
    word_freq = Counter(words)
    wordcloud = WordCloud(font_path='msyh.ttc', width=800, height=400, background_color='white', max_words=50)
    color_func = lambda word, **kwargs: random.choice(colors)
    wordcloud.generate_from_frequencies(word_freq)
    wordcloud.recolor(color_func=color_func)
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()

# 主函数
def main():
    text_path = 'C:\\Users\\zhesi\\Desktop\\定量.txt'
    stopwords_path = 'C:\\Users\\zhesi\\Desktop\\hit_stopwords.txt'
    pos_words_path = 'C:\\Users\\zhesi\\Desktop\\tsinghua_positive_gb.txt'
    neg_words_path = 'C:\\Users\\zhesi\\Desktop\\tsinghua_negative_gb.txt'

    # 加载文件
    text = load_file(text_path)
    stopwords = load_stopwords(stopwords_path)
    pos_words = load_stopwords(pos_words_path)
    neg_words = load_stopwords(neg_words_path)

    # 文本预处理
    words = preprocess_text(text, stopwords)

    # 情感分析
    sentiment_counts = analyze_sentiment(words, pos_words, neg_words)
    print("Sentiment counts:", sentiment_counts)

    # 生成词云
    create_wordcloud(words, ['#71BBEA', '#FE8D9F', '#FFAEC9'])

    plot_word_frequency(words)  # 绘制词频统计条形图
    analyze_and_plot_sentiments(words, pos_words, neg_words)  # 进行情感分析并绘制饼图

    # 进行主题建模
    dictionary = corpora.Dictionary([words])
    corpus = [dictionary.doc2bow(words)]
    ldamodel = models.LdaModel(corpus, num_topics=5, id2word=dictionary, passes=15)
    topics = ldamodel.print_topics(num_words=4)
    for topic in topics:
        print(topic)

if __name__ == '__main__':
    main()