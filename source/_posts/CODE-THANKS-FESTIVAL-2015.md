title: CODE THANKS FESTIVAL 2015
date: 2015-12-05 14:49:06
tags:
categories: 
- 競プロ
- CodeFestival
---

オンサイト参加資格が無いので，オープンに出ました．

http://code-thanks-festival-2015-open.contest.atcoder.jp/

### A

AからBに行くときの最短はabs(A - B)です．
この場合，100から-100に飛ぶなどということが出来ないので，この計算で大丈夫です．

``` cpp
int main() {
  int A, B; cin >>A >>B;
  cout <<(abs(A) + abs(A - B) + abs(B)) <<endl;
  return 0;
}
```

### B

全ての可能性を試します．
同じ数字を2回以上出力しないように注意して下さい．

``` cpp
typedef pair<int, int> P;
const int L = 2;

int main() {
  int A[L], B[L], C;
  REP(i, L) cin >>A[i];
  REP(i, L) cin >>B[i];
  cin >>C;
  set<P> S;
  set<int> ans;
  REP(i, L){
    REP(j, L){
      P p = P(min(A[i], B[j]), max(A[i], B[j]));
      if(S.find(p) != S.end()) continue;
      S.insert(p);
      if(p.first == C) ans.insert(p.second);
      if(p.second == C) ans.insert(p.first);
    }
  }
  cout <<ans.size() <<endl;
  for(int n : ans) cout <<n <<endl;
  return 0;
}
```

### C

生徒の数が少ないので，端から順に探していっても間に合います．

``` cpp
int solve(vector<int> &H, int X){
  REP(i, H.size()) if(H[i] > X) return i + 1;
  return H.size() + 1;
}

int main() {
  int N, X; cin >>N;
  vector<int> H(N);
  REP(i, N) cin >>H[i];
  cin >>X;
  cout <<solve(H, X) <<endl;
  return 0;
}
```

### D

既に今までのa = 0のクエリで得点を把握していれば，それを出力します．
そうでなければ

* p : (合計点 - 今分かっている得点)
* n : まだその人が得点を把握していない人の数

を使って，以下のように計算します．

* x点以上

max(0, p - ((n - 1) * 100))

得点の範囲を知りたい人c以外のn - 1人全てが100点だった場合の点を考えます．
0未満になってしまう場合はcが0点である可能性があるので，0とします．

* y点以下

pが100以上なら100，そうでなければp

n人全てが0点だった場合の点を考えます．


``` cpp
int N, sum = 0;
vector<int> S;
vector< vector<bool> > v;

void solve(int &b, int &c){
  int k = 0, cnt = 0;
  REP(i, N) if(v[b][i]) { k += S[i]; ++cnt; }
  int l = max(0, sum - k - (N - cnt - 1) * 100), h = (sum - k >= 100 ? 100 : sum - k);
  cout <<l <<" " <<h <<endl;
}

int main() {
  cin >>N;
  S = vector<int>(N);
  REP(i, N) { cin >>S[i]; sum += S[i]; }
  v = vector< vector<bool> >(N, vector<bool>(N, 0));
  REP(i, N) v[i][i] = 1;
  int M; cin >>M;
  int a, b, c;
  REP(i, M){
    cin >>a >>b >>c;
    --b; --c;
    if(a){
      if(v[b][c]) cout <<S[c] <<" " <<S[c] <<endl;
      else solve(b, c);
    }
    else v[b][c] = 1;
  }
  return 0;
}
```

### E

Sの中の，Tに含まれていない文字を全て消します．
消した結果の文字列S'の部分文字列にTがあればYES，そうでなければNOです．

``` cpp
const int ALPHA = 26;

bool check(string a, string &b){
  REP(i, (int)(a.length() - b.length()) + 1){
    if(a.substr(i, b.length()) == b) return true;
  }
  return false;
}

bool solve(string &a, string &b){
  stringstream ss;
  vector<bool> v(ALPHA, 0);
  for(char c : b) v[c - 'a'] = 1;
  for(char c : a) if(v[c - 'a']) ss << c;
  return check(ss.str(), b);
}

int main() {
  int Q; cin >>Q;
  string S, T;
  REP(q, Q){
    cin >>S >>T;
    cout <<(solve(S, T) ? "YES" : "NO") <<endl;
  }
  return 0;
}
```

-------

以下，本番中に解けず，解説を見て復習した問題です．

### F

両者が最善を尽くした場合，1の実がとられないように，可能な限り1に隣接していない実を食べるはずなので，最終的には1と，1に隣接した実のみが残ります．
この後，どちらが1の実を食べられるかどうかは，最初の実の数Nが偶数であるか奇数であるかで決定します．

``` cpp
int main() {
  int N, cnt = 0; cin >>N;
  REP(i, N - 1){
    int a, b; cin >>a >>b;
    if(a == 1 || b == 1) ++cnt;
  }
  if(cnt == 1) cout <<"A" <<endl;
  else{
    if(N % 2) cout <<"B" <<endl;
    else cout <<"A" <<endl;
  }
  return 0;
}
```

### G

色の種類が

> ci(1≦ci≦1,000,000,000)

と書いていて一瞬怯みますが，道の本数が

> M(1≦M≦80,000)

なので，色の種類は多くても80,000種類にしかならないことが分かります．

そのため，closedを `map<P, Int>` とでもmapで管理します．

余談ですが，私は本番で `map<P, Int>` ではなく`number_to_color` , `color_to_number`
という対応表みたいなmapを作ってダイクストラを書きましたが
これだとMLEで落ちてしまいました．

``` cpp
typedef long long int Int;
typedef pair<Int, Int> P;
typedef pair<P, Int> PP;
const Int INF = 1e18;

Int N, M;
vector< vector<PP> > v;

class C{
  public:
    Int now, cost, color;
    C(Int nn, Int tt, Int cc){ now = nn; cost = tt; color = cc; }
    bool operator > (const C &c) const { return cost > c.cost; }
};

Int solve(){
  priority_queue<C, vector<C>, greater<C> > open;
  open.push(C(0, 0, 1));
  map<P, Int> closed;
  while(!open.empty()){
    C c = open.top(); open.pop();
    if(c.cost > closed[P(c.now, c.color)]) continue;
    if(c.now == N - 1) return c.cost;
    REP(i, v[c.now].size()){
      Int to = v[c.now][i].first.first, color = v[c.now][i].first.second;
      Int cost = v[c.now][i].second;
      Int nextcost = c.cost + abs(c.color - color) + cost;
      if(closed.find(P(to, color)) == closed.end() || nextcost < closed[P(to, color)]){
        closed[P(to, color)] = nextcost;
        open.push(C(to, nextcost, color));
      }
    }
  }
  return INF;
}

int main() {
  cin >>N >>M;
  v = vector< vector<PP> >(N);
  REP(i, M){
    Int a, b, c, t; cin >>a >>b >>c >>t;
    --a; --b;
    v[a].push_back(PP(P(b, c), t));
    v[b].push_back(PP(P(a, c), t));
  }
  cout <<solve() <<endl;
  return 0;
}
```

### H

http://kmjp.hatenablog.jp/entry/2015/12/05/1100

ここが大変参考になります．
(しゃくとり法ができなさすぎて，ほぼ参考元そのまま)

累積和は自作ライブラリから．

long long intを使わないとWAになることに注意．

```
typedef long long int Int;
typedef pair<Int, Int> P;
typedef pair<P, P> PP;
#define Y first
#define X second
const Int MAX_H = 400;
const Int MAX_W = 400;
const Int N = 10;

Int H, W, K, ans, v[N][MAX_H][MAX_W], input[MAX_H][MAX_W];

void init(Int input[MAX_H][MAX_W]){
  memcpy(v[0], input, sizeof(v[0]));
  FOR(num, 1, N) REP(i, H) REP(j, W) v[num][i][j] = input[i][j] == num;
  REP(num, N){
    FOR(i, 1, H) v[num][i][0] += v[num][i - 1][0];
    FOR(i, 1, W) v[num][0][i] += v[num][0][i - 1];
    FOR(y, 1, H) FOR(x, 1, W) v[num][y][x] += v[num][y - 1][x] + v[num][y][x - 1] - v[num][y - 1][x - 1];
  }
}

Int calc(Int num, P a, P b){
  Int ret = v[num][b.Y][b.X];
  if(a.X - 1 >= 0) ret -= v[num][b.Y][a.X - 1];
  if(a.Y - 1 >= 0) ret -= v[num][a.Y - 1][b.X];
  if(a.X - 1 >= 0 && a.Y - 1 >= 0) ret += v[num][a.Y - 1][a.X - 1];
  return ret;
}

Int solve(){
  Int res = 0;
  REP(y1, H - 2){
    REP(x1, W - 2){
      Int n = y1 + 2;
      for(Int x2 = W - 1; x2 >= x1 + 2; --x2){
        while(n < H && calc(0, P(y1, x1), P(n, x2)) <= K) ++n;
        FOR(y2, n, H){
          Int num = calc(0, P(y1, x1), P(y2, x2));
          if(num - K > 9) break;
          res += calc(num - K, P(y1 + 1, x1 + 1), P(y2 - 1, x2 - 1));
        }
      }
    }
  }
  return res;
}

int main() {
  cin >>H >>W >>K;
  REP(i, H){
    REP(j, W){
      char c; cin >>c;
      input[i][j] = c - '0';
    }
  }
  init(input);
  cout <<solve() <<endl;
  return 0;
}
```
