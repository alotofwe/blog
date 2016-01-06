title: Rspecのバリデーション周りのテストをshared_examples_forですっきりさせる
date: 2016-01-01 17:41:46
tags:
- rspec
categories:
- application
- rails
---

Rspecの `shared_examples_for` を使うと，同じようなテストを使い回すことができます．  

よく書くテストというと，Railsでは特にバリデーション周りがあるかと思うので，  
これを使ってうまく使いまわせるようなテストを書くと，テストが長くならずに楽かと思います．

バリデーション周りだと

* 同じ制約を複数のカラムに適用させている
  - カラム名が違うのでコピーしてカラム名を書き換え
  - 文字数等の指定が違うのでコピーして値を書き換え

という問題で，テストが長くなりがちかと思いますが  
以下の書き方をすると，1テスト1行で書くことができます．

## shared_examples_for

まず，使い回すテストを書きます．  

前述の，各テストについての微妙な差異は  
shared_examples_forに引数を与えることができるという点を利用します．

前提として，FactoryGirlで  
引数を取らないような `クラス名 = ファクトリ名` の定義があるものとします．

* 例) 値が空の時に保存することができない

```
shared_examples_for 'validate presence' do |column_name|
  let(:factory_name) { described_class.name.underscore }
  let(:record)       { build(factory_name) }

  context "#{column_name}が空の時" do
    before { record[column_name] = "" }

    it '保存することができない' do
      expect(record).to be_invalid
    end
  end
end
```

* 例) 値が指定された長さの時に保存することができない

```
shared_examples_for 'validate length' do |column_name, length|
  let(:factory_name) { described_class.name.underscore }
  let(:record)       { build(factory_name) }

  context "#{column_name}が#{length}文字の時" do
    before { record[column_name] = "a" * length.to_i }

    it '保存することができない' do
      expect(record).to be_invalid
    end
  end
end
```

引数として，必ず対象のカラム名を取るように  
また他の値が必要であれば，第二引数以降にそれも取るようにします．

これを， `spec/shared/` 以下にでも保存しておいて下さい．

## it_behaves_like

実際にテストを書くべき箇所で，上のテストを使いまわします．

例えば，

* nameというカラムが空の時は保存されない
* nameというカラムの長さが256の時は保存されない

ということを確認したい時は

```
it_behaves_like 'validate presence', :name
it_behaves_like 'validate length',   :name, 256
```

と書くことができます．

また，違うカラムについて同様のテストを試すこともできます．

```
it_behaves_like 'validate length', :description, 256
```

こう書くと見た目にも分かりやすく，長さが変更された際も修正が楽かと思います．
