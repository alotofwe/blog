title: Advent of Code 2015
date: 2015-12-10 00:43:05
tags:
categories:
- 競プロ
- その他
---

http://adventofcode.com/

やります．
Part Twoは別記事で．

### Day 1

`++` か `--` するだけ．

```
int main() {
  string s; cin >>s;
  int f = 0;
  for(char c : s){
    if(c == '(') f++;
    else f--;
  }
  cout <<f <<endl;
  return 0;
}
```

### Day 2

計算するだけ．
scanf使えばよかった感．

```
vector<int> split(string &str, char delim){
  stringstream ss(str);
  string tmp;
  vector<int> res;
  while(getline(ss, tmp, delim)) res.push_back(atoi(tmp.c_str()));
  return res;
}

int main() {
  int ans = 0;
  string s;
  while(cin >>s){
    vector<int> v = split(s, 'x');
    int m = 1e9;
    REP(i, v.size()){
      ans += v[i] * v[(i + 1) % (int)v.size()] * 2;
      m = min(m, v[i] * v[(i + 1) % (int)v.size()]);
    }
    ans += m;
  }
  cout <<ans <<endl;
  return 0;
}
```

### Day 3

座標を(x, y)で表現し，最初にいるところを(0, 0)とする．
その後，移動しながら現在地をSetに入れていく．

```
typedef complex<int> P;
namespace std {
  bool operator < (const P& a, const P& b) {
    return real(a) != real(b) ? real(a) < real(b) : imag(a) < imag(b);
  }
}
 
int main() {
  P p = P(0, 0);
  int ans = 1;
  string s; cin >>s;
  map<char, P> m;
  m['^'] = P(0, -1);
  m['>'] = P(1, 0);
  m['v'] = P(0, 1);
  m['<'] = P(-1, 0);
  set<P> S;
  S.insert(P(0, 0));
  for(char c : s){
    p += m[c];
    S.insert(p);
  }
  cout <<S.size() <<endl;
  return 0;
}
```

### Day 4

適当に，0から順にくっつけてハッシュ値を調べる．
C++でハッシュ値を求めるのが面倒だったので，rubyで．

```
require 'digest/md5'
N = 10000000

def solve key
  N.times do |t|
    md5 = Digest::MD5.hexdigest("#{key}#{t.to_s}")
    return t if md5[0, 5] == "00000"
  end
  return "error"
end

puts solve('ckczppom')
```

### Day 5

3つの条件をチェックしていく．

私は1文字の部分文字列と2文字の部分文字列を `map<string, int>` に入れ
部分文字列とその数を数えていった．

```
int check(string &s){
  map<string, int> M;
  for(int i = 0; i < (int)s.length(); i++) M[s.substr(i, 1)] += 1;
  for(int i = 0; i < (int)s.length() - 1; i++) M[s.substr(i, 2)] += 1;
  int x = 0, y = 0, z = 0;
  string vowels[] = {"a", "e", "i", "o", "u"};
  for(string vowel : vowels) x += M[vowel];
  for(int c = 'a'; c <= 'z'; c++){
    stringstream ss;
    ss << (char)c << (char)c;
    y += M[ss.str()];
  }
  string invalid[] = {"ab", "cd", "pq", "xy"};
  for(string inv : invalid) z += M[inv];
  return x >= 3 && y >= 1 && z == 0;
}
 
int main() {
  string s;
  int ans = 0;
  while(cin >>s) ans += check(s);
  cout <<ans <<endl;
  return 0;
}
```

### Day 6

2次元配列をつくって，つけたり消したりしていく．
toggleの時だけ単語数が減って不便なので，テストケースを `turn toggle` に書きかえた．

```
const int H = 1010;
const int W = 1010;

bool v[H][W];

void check(int type, int &x1, int &y1, int &x2, int &y2){
  FOR(x, x1, x2 + 1){
    FOR(y, y1, y2 + 1){
      if(type == 0) v[x][y] = 0;
      else if(type == 1) v[x][y] = 1;
      else v[x][y] = !v[x][y];
    }
  }
}

int main() {
  char s[10];
  int x1, y1, x2, y2;
  while(scanf("turn %s %d,%d through %d,%d\n", s, &x1, &y1, &x2, &y2) != EOF){
    int type = 2;
    if(strcmp(s, "off") == 0) type = 0;
    if(strcmp(s, "on") == 0) type = 1;
    check(type, x1, y1, x2, y2);
  }
  int ans = 0;
  REP(i, H) REP(j, W) ans += v[i][j];
  cout <<ans <<endl;
  return 0;
}

```

### Day 7

16bitの符号なし整数なので， `unsigned short` を使う．

変数と数字を対応させるmapを用意する．
0~65534を特別扱いするのが面倒なので， `(string) n = (unsigned short) n` として突っ込んでおく．

最初に，全ての式をqueueに入れる．

順に取り出し，全ての変数が分かっている式であれば，mapに値を反映させる．
そうでなければqueueに入れなおす．

```
vector<string> split(string &str, char delim){
  stringstream ss(str);
  string tmp;
  vector<string> res;
  while(getline(ss, tmp, delim)) res.push_back(tmp);
  return res;
}

int main() {
  map<string, unsigned short> M;
  REP(i, 65535){ stringstream ss; ss << i; M[ss.str()] = i; }

  queue<string> Q;
  string input;
  while(getline(cin, input)) Q.push(input);

  while(!Q.empty()){
    string str = Q.front(); Q.pop();
    vector<string> v = split(str, ' ');
    if(v.size() == 3 && M.find(v[0]) != M.end()){
      M[v[2]] = M[v[0]];
    } else if(v.size() == 4 && M.find(v[1]) != M.end()){
      M[v[3]] = ~M[v[1]];
    } else if(v.size() == 5 && v[1] == "AND" && M.find(v[0]) != M.end() && M.find(v[2]) != M.end()){
      M[v[4]] = M[v[0]] & M[v[2]];
    } else if(v.size() == 5 && v[1] == "OR" && M.find(v[0]) != M.end() && M.find(v[2]) != M.end()){
      M[v[4]] = M[v[0]] | M[v[2]];
    } else if(v.size() == 5 && v[1] == "LSHIFT" && M.find(v[0]) != M.end() && M.find(v[2]) != M.end()){
      M[v[4]] = M[v[0]] << M[v[2]];
    } else if(v.size() == 5 && v[1] == "RSHIFT" && M.find(v[0]) != M.end() && M.find(v[2]) != M.end()){
      M[v[4]] = M[v[0]] >> M[v[2]];
    } else{
      Q.push(str);
    }
  }
  cout <<M["a"] <<endl;
  return 0;
}
```

### Day 8

エスケープ文字を頑張ってスキップする．
両側の"もカウントしない．

```
int countChars(string &s){
  int res = 0;
  for(int i = 1; i < (int)s.length() - 1; ++i, ++res){
    if(s[i] == '\\'){
      if(s[i + 1] == 'x') i += 3;
      else i += 1;
    }
  }
  return res;
}

int main() {
  int a = 0, b = 0;
  string s;
  while(cin >>s){
    a += (int)s.length();
    b += countChars(s);
  }
  cout <<a - b <<endl;
  return 0;
}
```

### Day 9

入力の形式が分かりづらいので，人名を数字になおして距離行列にする．
人は8種類しか存在しないので，何も考えない深さ優先探索で十分間に合う．

```
const int INF = 1e9;

int N;
vector< vector<int> > v;

vector< vector<int> > prepare(){
  vector<string> ss; string s;
  while(getline(cin, s)) ss.push_back(s);

  char a[100], b[100]; int d;
  set<string> S;
  REP(i, ss.size()){
    sscanf(ss[i].c_str(), "%s to %s = %d\n", a, b, &d);
    S.insert(a); S.insert(b);
  }

  map<string, int> M; int num = 0;
  for(string str : S) M[str] = num++;

  vector< vector<int> > res(S.size(), vector<int>(S.size(), INF));
  REP(i, res.size()) res[i][i] = 0;
  REP(i, ss.size()){
    sscanf(ss[i].c_str(), "%s to %s = %d\n", a, b, &d);
    res[M[a]][M[b]] = d;
    res[M[b]][M[a]] = d;
  }
  return res;
}

int solve(int now, int cost, vector<bool> &visited){
  visited[now] = 1;
  int res = INF;
  REP(i, N) if(!visited[i]) res = min(res, solve(i, cost + v[now][i], visited));
  visited[now] = 0;
  return res == INF ? cost : res;
}

int main() {
  v = prepare();
  N = v.size();
  vector<bool> visited(N, 0);
  int ans = INF;
  REP(i, N) ans = min(ans, solve(i, 0, visited));
  cout <<ans <<endl;
  return 0;
}
```

### Day 10

1つ前の数字を見て，一致しているかどうかによって処理を変える．

```
string solve(string &s){
  stringstream ss;
  int cnt = 1;
  FOR(i, 1, s.length()){
    if(s[i] != s[i - 1]){
      ss << cnt << s[i - 1];
      cnt = 1;
    } else ++cnt;
  }
  ss << cnt << s[s.length() - 1];
  return ss.str();
}

int main() {
  string s = "1113222113";
  REP(i, 40) s = solve(s);
  cout <<s.length() <<endl;
  return 0;
}
```

### Day 11

C++では文字に1を足すと次の文字になるので，それを利用する．

```
void inc(string &s){
  s[s.size() - 1]++;
  for(int i = s.size() - 1; i > 0; --i) if(s[i] > 'z'){ s[i] = 'a'; s[i - 1]++; }
}

bool check(string &s){
  REP(i, s.size()) if(s[i] == 'i' || s[i] == 'o' || s[i] == 'l') return false;
  int a = 0, b = 0;
  REP(i, (int)s.size() - 2) if(s[i] + 1 == s[i + 1] && s[i + 1] + 1 == s[i + 2]) ++a;
  REP(i, (int)s.size() - 1) if(s[i] == s[i + 1]) { ++b; ++i; } ;
  return a > 0 && b > 1;
}

int main() {
  string s = "hepxcrrq";
  while(!check(s)) inc(s);
  cout <<s <<endl;
  return 0;
}
```
