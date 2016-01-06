title: RailsでSidekiqのログをLTSVで出力する
date: 2015-12-03 23:42:56
tags:
- sidekiq
- ltsv
categories: 
- application
- rails
---

これ、置いておきますね。

``` text Gemfile
gem 'ltsv'
gem 'sidekiq'
```

``` ruby config/initializers/sidekiq.yml

class SidekiqLtsvFormatter < Sidekiq::Logging::Pretty
  def call(severity, time, _, message)
    "#{LTSV.dump({
      time: time.utc.iso8601(3),
      pid: ::Process.pid,
      tid: Thread.current.object_id.to_s(36),
      context: context,
      level: severity,
      message: message
    })}\n"
  end
end

Sidekiq.logger.formatter = SidekiqLtsvFormatter.new
```

だけではあれなので、少し説明をします。

------

Sidekiq gemのログは

https://github.com/mperham/sidekiq/blob/master/lib/sidekiq/logging.rb#L10-L13

このcallの返り値を出力したものになります。

よって、このcallを何らかの方法で置き換えてあげれば、出力する内容を変えることができます。  
例えば

``` ruby
def call(severity, time, _, message)
  "ピエールおはよ〜"
end
```

とすると、ログに `ピエールおはよ〜ピエールおはよ〜ピエールおはよ〜 ...`と出力されるということです。  
callで改行を行わないと、ログが改行されずに出力されることに注意です。

データをLTSVに成形するために、 `ltsv` というgemを使っています。  
使い方の詳細は[ltsvのREADME](https://github.com/condor/ltsv/blob/master/README.md)参照です。

最後に、Sidekiq.logger.formatterを自作のもので置き換えてあげれば終了です。  

sidekiqの再起動が必要かもしれません。変わらなければ一度再起動をしてみてください。
