#include "BAGEL.h"

double fastncdf_pos(double x){
  if(x >= fastncdf_max)  return 1.0;

  const int i = (int)(x * fastncdf_hinv);
  const double w = (x - fastncdf_x[i]) * fastncdf_hinv;
  return w * fastncdf_y[i + 1] + (1.0 - w) * fastncdf_y[i];
}
double fastncdf(double x){
  if(x < 0)
    return 1.0 - fastncdf_pos(-x);

  return fastncdf_pos(x);
}

arma::mat rowsome(arma::mat x,arma::rowvec ind){
  //index starts from 0
  int C = x.n_cols;
  int R = ind.n_cols;
  arma::mat ans(R,C);
  for(int i=0; i<R; i++){
    ans.row(i) = x.row(ind(i));
  }
  return(ans);
}
arma::mat colsome(arma::mat x,arma::rowvec ind){
  //index starts from 0
  int R = x.n_rows;
  int C = ind.n_cols;
  arma::mat ans(R,C);
  for(int i=0; i<C; i++){
    ans.col(i) = x.col(ind(i));
  }
  return(ans);
}
int rmunoim(arma::rowvec probs) {
  //this function returns a int, not a vector
  //rmultinom(1,1,probs)
  int k = probs.n_cols;
  IntegerVector ans(k);
  int r=0;
  rmultinom(1, probs.begin(), k, ans.begin());
  for(int i=0; i<k; i++){
    if(ans(i)==1){
      r=i+1;
    }
  }
  return r;
}
int sampleint(int n){
  arma::rowvec beta(n);
  arma::mat temp(1,1);
  temp = 1.0/n;
  beta = arma::repmat(temp,1,n);
  int ans = rmunoim(beta);
  return ans;
}
arma::rowvec getind(arma::rowvec x, double c){
  int n = x.n_cols;
  arma::rowvec ans(n);
  int flag = 0;
  for(int i=0; i<n; i++){
    if(x(i)==c){
      ans(flag) = i;
      flag++;
    }
  }
  ans.reshape(1,flag);
  return(ans);
}
arma::rowvec whichin(arma::rowvec data,arma::rowvec set){
  //index starts from 0
  int C = set.n_cols;
  int N = data.n_cols;
  arma::rowvec ans(N);

  int flag = 0;
  for(int i=0; i<N; i++){
    for(int j=0; j<C; j++){
      if(data(i)==set(j)){
        ans(flag) = i;
        flag++;
        break;
      }
    }
  }
  ans.reshape(1,flag);
  return(ans);
}
arma::rowvec n_to_m(int n, int m){
  int l = m-n + 1;
  arma::rowvec ans(l);
  for(int i=n; i<=m; i++){
    ans(i-n)=i;
  }
  return (ans);
}
arma::rowvec repp(double x, int l){
  arma::rowvec ans(l);
  ans.fill(x);
  return (ans);
}
arma::mat mymatrix2(arma::vec data, int nrow, int ncol, bool byrow = true){
  //byrow= T
  arma::mat ans(1,nrow*ncol);
  ans = data;

  if(byrow){
    ans.reshape(ncol, nrow);
    return ans.t();
  }else{
    ans.reshape(nrow, ncol);
    return ans;
  }
}
arma::mat put_rows(arma::mat& data, arma::rowvec ind, arma::mat data0){
  //byrow= T
  int l = ind.n_cols;
  for(int i=0; i<l; i++){
    data.row(ind(i)) = data0.row(i);
  }
  return data;
}

arma::mat compute_mu(int H, arma::rowvec e,
                          arma::cube alpha, arma::cube gamma,
                          arma::cube R, arma::cube beta,
                          arma::mat Z, arma::mat data,
                          int D, int Q, int S, int B){
  int NN = data.n_rows;
  arma::mat mu(NN,Q);
  //arma::colvec et = e.t();
  arma::rowvec pt_id(NN);
  arma::rowvec tmp_ind(NN);
  arma::mat Xk(NN, S);
  arma::mat Zkt(D,NN);
  arma::mat bk(NN, B);
  arma::mat alphakt(D*Q, S);
  arma::mat gammakt(D*Q,B);
  arma::mat Rk(Q,D);
  arma::mat betak(Q, 1+S);
  arma::mat betaZ(NN,Q);
  arma::mat BX(NN,Q);
  arma::mat zalpha(D*Q,NN);
  arma::mat bxbi(D*Q,NN);
  arma::mat BB(Q,D);
  arma::mat BBR(Q,D);
  arma::mat tmp(D,1);
  int ll,LL;

  for(int k=1; k<=H; k++){
    ll = getind(e, k).n_cols;
    pt_id.reshape(1,ll);
    pt_id = colsome(unique(data.col(0).t()), getind(e,k));

    LL = whichin(data.col(0).t(), pt_id).n_cols;
    tmp_ind.reshape(1,LL);
    tmp_ind = whichin(data.col(0).t(), pt_id);


    Xk.reshape(LL,S);
    Zkt.reshape(D,LL);
    bk.reshape(LL,B);

    Xk = colsome(rowsome(data, tmp_ind), n_to_m(2, 2+S-1));
    Zkt = rowsome(Z,tmp_ind).t();
    bk = colsome(rowsome(data, tmp_ind), n_to_m(2+S, 1+S+B));

    alphakt = alpha.slice(k-1).t();
    gammakt = gamma.slice(k-1).t();
    Rk = R.slice(k-1);
    betak = beta.slice(k-1);

    betaZ.reshape(LL,Q);
    BX.reshape(LL,Q);
    betaZ = join_rows(repp(1,LL).t(), Xk) * betak.t();

    zalpha.reshape(D*Q,LL);
    bxbi.reshape(D*Q,LL);

    zalpha =  alphakt * Xk.t();
    bxbi =  gammakt * bk.t();

    for(int i=0; i<LL; i++){
      BB = mymatrix2( zalpha.col(i) + bxbi.col(i), Q, D);
      BBR = BB % Rk;
      tmp = BBR * Zkt.col(i);
      BX = put_rows(BX, n_to_m(i,i), tmp.t());

    }
    mu = put_rows(mu, tmp_ind, betaZ+BX);
  }
  return mu;
}

arma::mat compute_muk2(arma::mat& alphakt, arma::mat gammakt,
                           arma::mat& Rk, arma::mat& betak,
                           arma::mat& Zkt, arma::mat& datak,
                           int D, int Q, int S, int B){
  int NN = datak.n_rows;
  arma::mat mu(NN,Q);


  arma::mat Xk(NN, S);
  arma::mat bk(NN, B);


  arma::mat betaZ(NN,Q);
  arma::mat BX(NN,Q);
  arma::mat zalpha(D*Q, NN);
  arma::mat bxbi(D*Q, NN);
  arma::mat BB(Q,D);
  arma::mat BBR(Q,D);
  arma::mat tmp(Q,1);


  Xk = colsome(datak, n_to_m(2, 2+S-1));
  bk = colsome(datak, n_to_m(2+S, 1+S+B));

  //AZ = join_rows(repp(1,NN).t(),Zk) * Ak.t();
  betaZ = join_rows(repp(1,NN).t(), Xk) * betak.t();

  //zalpha = Zk * alphak;
  zalpha =  alphakt * Xk.t();
  bxbi =  gammakt * bk.t();



  for(int i=0; i<NN; i++){
    BB = mymatrix2( zalpha.col(i) + bxbi.col(i), Q, D);
    BBR = BB % Rk;
    tmp = BBR * Zkt.col(i);
    BX = put_rows(BX, n_to_m(i,i), tmp.t());

  }
  //mu[tmp_ind,] <- AZ +BX
  mu = betaZ+BX;
  return mu;

}

arma::colvec compute_muk_j2(arma::mat alphakt, arma::mat gammakt,
                                arma::mat& Rk, arma::mat& betak,
                                arma::mat& Zkt, arma::mat& datak,  int j,
                                int D, int Q, int S, int B){
  int NN = datak.n_rows;
  arma::colvec mu(NN);
  arma::rowvec ind(D);
  ind = n_to_m(j*D, j*D+D-1);

  arma::mat Xk(NN, S);
  arma::mat bk(NN, B);

  arma::colvec betaZ(NN);
  arma::colvec BX(NN);
  arma::mat zalpha(D,NN);
  arma::mat bxbi(D,NN);
  arma::colvec BB(D);
  arma::colvec BBR(D);
  arma::mat tmp(1,1);


  Xk = colsome(datak, n_to_m(2, 2+S-1));
  bk = colsome(datak, n_to_m(2+S, 1+S+B));

  betaZ = join_rows(repp(1,NN).t(), Xk) * betak.row(j).t();

  zalpha =  rowsome(alphakt, ind) * Xk.t();
  bxbi =  rowsome(gammakt,ind) * bk.t();


  for(int i=0; i<NN; i++){
    BB = zalpha.col(i) + bxbi.col(i);
    BBR = BB % Rk.row(j).t();
    tmp = BBR.t() * Zkt.col(i);
    BX(i) = tmp(0,0);
  }
  //mu[tmp_ind,] <- AZ +BX
  mu = betaZ+BX;
  return mu;

}

arma::colvec rmvnorm_col(arma::colvec mean, arma::mat var){
  int n = mean.n_rows;
  arma::colvec ans(n);
  arma::colvec Z(n);
  arma::mat R(n,n);
  arma::mat tmp(n,1);
  R = chol(var);
  for(int i=0; i<n; i++){
    Z(i) = R::rnorm(0,1);
  }
  tmp = R.t() * Z;
  ans = mean + tmp;
  //mu + R^TZ
  return (ans);

}

arma::rowvec whichno(arma::colvec v, double x){
  int n = v.n_rows;
  arma::rowvec ans(n);
  int flag = 0;
  for(int i=0; i<n; i++){
    if(v(i)!=x){
      ans(flag) = i;
      flag++;
    }
  }
  ans.reshape(1,flag);
  return(ans);
}

arma::mat eye(int n){
  arma:: mat ans(n,n);
  ans.eye();
  return (ans);
}

arma::mat put_col_somerows(arma::mat A, arma::rowvec ind, int j, arma::colvec b){
  int N = ind.n_cols;
  for(int i=0; i<N; i++){
    A(ind(i), j) = b(i);
  }
  return A;
}


List update_alpha(int H, arma::rowvec& e,
                     arma::cube alpha, arma::cube& gamma,
                     arma::cube& R, arma::cube& beta,
                     arma::mat& Z, arma::mat& data,
                     int D, int Q, int S, int B,
                     arma::mat mu, arma::mat Y,
                     double sigma_square, double sigma_square_alpha){
  List ans;
  arma::cube alpha_new = alpha;
  int NN = data.n_rows;

  arma::rowvec pt_id(NN);
  arma::rowvec tmp_ind(NN);
  arma::mat alphak_new(S, D*Q);
  arma::mat alphak0(S, D*Q);
  arma::mat gammakt(Q*D,B);
  arma::mat Rk(Q,D);
  arma::mat betak(Q, 1+S);
  arma::mat Yk(NN, Q);
  arma::mat datak(NN, 2+S+B);
  arma::mat Zkt(D,NN);
  arma::mat datakin(NN, 2+S+B);
  arma::mat Zktin(D,NN);
  arma::mat muk(NN, Q);
  arma::colvec muk0(NN);
  arma::colvec muk_change(NN);


  arma::colvec diff(NN);
  arma::rowvec ind(NN);
  arma::colvec Yalpha(NN);
  arma::mat Xk(NN, S);
  arma::mat covalpha(S,S);
  int ll, LL, l;

  for(int k=0; k<H; k++){
    alphak_new = alpha.slice(k);

    ll = getind(e, k+1).n_cols;
    pt_id.reshape(1,ll);
    pt_id = colsome(unique(data.col(0).t()), getind(e,k+1));

    LL = whichin(data.col(0).t(), pt_id).n_cols;
    tmp_ind.reshape(1,LL);
    tmp_ind = whichin(data.col(0).t(), pt_id);

    Yk.reshape(LL, Q);
    datak.reshape(LL, 2+S+B);
    Zkt.reshape(D,LL);
    muk.reshape(LL,Q);

    Yk = rowsome(Y, tmp_ind);
    datak = rowsome(data, tmp_ind);
    Zkt = rowsome(Z, tmp_ind).t();
    muk = rowsome(mu, tmp_ind);

    gammakt = gamma.slice(k).t();
    Rk = R.slice(k);
    betak = beta.slice(k);


    for(int i=0; i<Q; i++){
      for(int j=0; j<D; j++){
        if(Rk(i,j)==0){
          for(int tt=0; tt<S; tt++){
            alphak_new(tt, i*D+j) = R::rnorm(0,std::sqrt(sigma_square_alpha));
          }
          continue;
        }

        l = whichno(Zkt.row(j).t(),0).n_cols;
        if(l==0){
          for(int tt=0; tt<S; tt++){
            alphak_new(tt, i*D+j) = R::rnorm(0,std::sqrt(sigma_square_alpha));
          }
          continue;
        }

        datakin.reshape(l,2+S+B);
        Zktin.reshape(l,D);

        ind.reshape(1,l);
        ind = whichno(Zkt.row(j).t(),0);

        datakin = rowsome(datak,ind);
        Zktin = colsome(Zkt, ind);

        alphak0 = alphak_new;
        alphak0.col(i*D + j).fill(0);
        muk0.reshape(l,1);
        muk_change.reshape(l,1);
        muk0 = compute_muk_j2(alphak0.t(), gammakt, Rk, betak, Zktin, datakin,i, D, Q, S, B);

        Yalpha.reshape(l,1);
        Yalpha = rowsome(Yk.col(i), ind) - muk0 ;
        Xk.reshape(l, S);
        Xk =colsome(rowsome(datak, ind), n_to_m(2, 1+S)) ;
        covalpha = inv(Xk.t() * Xk/sigma_square + eye(S)/sigma_square_alpha);
        alphak_new.col(i*D+j) = rmvnorm_col(covalpha * Xk.t() * Yalpha/sigma_square, covalpha);

        muk_change = compute_muk_j2(alphak_new.t(), gammakt, Rk, betak,Zktin, datakin,
                                        i, D, Q, S,  B);
        muk = put_col_somerows(muk, ind, i, muk_change);

      }
    }
    alpha_new.slice(k) = alphak_new;
    mu = put_rows(mu, tmp_ind, muk);
  }
  ans["mu"] = mu;
  ans["alpha"] = alpha_new;
  return ans;
}


arma::mat makeP(int B, int deg = 2){
  arma::mat ans(B,B);
  ans.eye();
  if(deg>0){
    for(int i=0; i<deg; i++){
      ans = diff(ans);
    }
  }
  ans = ans.t() * ans;
  return (ans);
}

List update_gamma(int H, arma::rowvec& e,
                      arma::cube& alpha, arma::cube gamma,
                      arma::cube& R, arma::cube& beta,
                      arma::mat& Z, arma::mat& data,
                      int D, int Q, int S, int B,
                      arma::mat mu, arma::mat Y,
                      double sigma_square, double sigma_square_gamma){
  List ans;
  arma::cube gamma_new = gamma;
  int NN = data.n_rows;

  arma::rowvec pt_id(NN);
  arma::rowvec tmp_ind(NN);
  arma::mat alphakt(D*Q,S);
  arma::mat gammak_new(B, D*Q);
  arma::mat gammak0(B, D*Q);
  //arma::mat gammak(M,q*p);
  arma::mat Rk(Q,D);
  arma::mat betak(Q, 1+S);
  arma::mat Yk(NN, Q);
  arma::mat datak(NN, 2+S+B);
  arma::mat Zkt(D,NN);
  arma::mat datakin(NN, 2+S+B);
  arma::mat Zktin(D,NN);
  arma::mat muk(NN, Q);
  arma::colvec muk0(NN);
  arma::colvec muk_change(NN);


  arma::colvec diff(NN);
  arma::rowvec ind(NN);
  arma::colvec Ygamma(NN);
  arma::mat Xgamma(NN, B);
  arma::mat covgamma(B,B);
  int l, LL, ll;

  for(int k=0; k<H; k++){
    gammak_new = gamma.slice(k);

    ll = getind(e, k+1).n_cols;
    pt_id.reshape(1,ll);
    pt_id = colsome(unique(data.col(0).t()), getind(e,k+1));

    LL = whichin(data.col(0).t(), pt_id).n_cols;
    tmp_ind.reshape(1,LL);
    tmp_ind = whichin(data.col(0).t(), pt_id);

    Yk.reshape(LL, Q);
    datak.reshape(LL, 2+S+B);
    Zkt.reshape(D,LL);
    muk.reshape(LL,Q);

    Yk = rowsome(Y, tmp_ind);
    datak = rowsome(data, tmp_ind);
    Zkt = rowsome(Z, tmp_ind).t();
    muk = rowsome(mu, tmp_ind);



    alphakt = alpha.slice(k).t();
    //gammak = gamma.slice(k);
    Rk = R.slice(k);
    betak = beta.slice(k);

    for(int i=0; i<Q; i++){
      for(int j=0; j<D; j++){
        if(Rk(i,j)==0){
          gammak_new.col(i*D+j) = rmvnorm_col(repp(0, B).t(), sigma_square_gamma * pinv(makeP(B))+eye(B)/10000);
          continue;
        }

        l = whichno(Zkt.row(j).t(),0).n_cols;
        if(l==0){
          gammak_new.col(i*D+j) = rmvnorm_col(repp(0, B).t(), sigma_square_gamma * pinv(makeP(B))+eye(B)/10000);
          continue;
        }

        datakin.reshape(l,2+S+B);
        Zktin.reshape(l,D);

        ind.reshape(1,l);
        ind = whichno(Zkt.row(j).t(),0);

        datakin = rowsome(datak,ind);
        Zktin = colsome(Zkt, ind);

        gammak0 = gammak_new;
        gammak0.col(i*D + j).fill(0);
        muk0.reshape(l,1);
        muk_change.reshape(l,1);

        muk0 = compute_muk_j2( alphakt, gammak0.t(), Rk, betak, Zktin, datakin,
                                   i,D, Q, S, B);

        Ygamma.reshape(l,1);
        Ygamma = rowsome(Yk.col(i), ind) - muk0;
        Xgamma.reshape(l, B);
        Xgamma = colsome(rowsome(datak, ind),  n_to_m(2+S, 1+S+B));

        covgamma = pinv(Xgamma.t() * Xgamma/sigma_square + (makeP(B))/sigma_square_gamma);
        gammak_new.col(i*D+j)  = rmvnorm_col(covgamma * Xgamma.t() * Ygamma/sigma_square, covgamma+eye(B)/1000000);

        muk_change = compute_muk_j2(alphakt, gammak_new.t(), Rk, betak, Zktin, datakin,
                                        i,D, Q, S, B);

        muk = put_col_somerows(muk, ind, i, muk_change);
      }
    }
    gamma_new.slice(k) = gammak_new;
    mu = put_rows(mu, tmp_ind, muk);
  }
  ans["mu"] = mu;
  ans["gamma"] = gamma_new;
  return ans;
}


arma::cube update_beta(int H, arma::rowvec& e,
                         arma::cube& alpha, arma::cube& gamma,
                         arma::cube& R, arma::cube beta,
                         arma::mat& Z, arma::mat& data,
                         int D, int Q, int S, int B, arma::mat Y,
                         double sigma_square, double sigma_square_beta){
  List ans;
  arma::mat betakt(1+S,Q);
  int NN = data.n_rows;
  arma::rowvec pt_id(NN);
  arma::rowvec tmp_ind(NN);
  arma::mat Xk(NN, S);
  arma::mat Zkt(D,NN);
  arma::mat bk(NN, B);
  arma::mat alphakt(D*Q, S);
  arma::mat gammakt(Q*D,B);
  arma::mat Rk(Q,D);
  arma::mat betaZ(NN,Q);
  arma::mat BX(NN,Q);
  arma::mat zalpha(D*Q,NN);
  arma::mat bxbi(D*Q,NN);
  arma::mat BB(Q,D);
  arma::mat BBR(Q,D);
  arma::mat tmp(Q,1);
  arma::mat Xa(NN, 1+S);
  arma::mat covbeta(1+S, 1+S);
  arma::colvec meanbeta(1+S);


  int ll,LL;

  beta.fill(0);
  for(int k=1; k<=H; k++){
    betakt.fill(0);

    ll = getind(e, k).n_cols;
    pt_id.reshape(1,ll);
    pt_id = colsome(unique(data.col(0).t()), getind(e,k));

    LL = whichin(data.col(0).t(), pt_id).n_cols;
    tmp_ind.reshape(1,LL);
    tmp_ind = whichin(data.col(0).t(), pt_id);


    Xk.reshape(LL,S);
    Zkt.reshape(D,LL);
    bk.reshape(LL,B);

    Xk = colsome(rowsome(data, tmp_ind), n_to_m(2, 2+S-1));

    Zkt = rowsome(Z,tmp_ind).t();
    bk = colsome(rowsome(data, tmp_ind), n_to_m(2+S, 1+S+B));

    alphakt = alpha.slice(k-1).t();
    gammakt = gamma.slice(k-1).t();
    Rk = R.slice(k-1);

    //BX <- matrix(NA, nrow = LL, ncol = q)
    BX.reshape(LL,Q);

    //wpsi <- wk %*% psik
    //zalpha <- Zk %*% alphak
    zalpha.reshape(D*Q,LL);
    bxbi.reshape(D*Q,LL);

    zalpha =  alphakt * Xk.t();
    bxbi =  gammakt * bk.t();


    for(int i=0; i<LL; i++){
      BB = mymatrix2( zalpha.col(i) + bxbi.col(i), Q, D);
      BBR = BB % Rk;
      tmp = BBR * Zkt.col(i);
      BX = put_rows(BX, n_to_m(i,i), tmp.t());
    }

    betaZ.reshape(LL,Q);
    Xa.reshape(LL,1+S);
    betaZ = rowsome(Y,tmp_ind) - BX;

    Xa = join_rows(repp(1,LL).t(), Xk);

    covbeta = inv(Xa.t() * Xa/sigma_square + eye(1+S)/sigma_square_beta);
    for(int j=0; j<Q; j++){
      meanbeta = covbeta* Xa.t() * betaZ.col(j)/sigma_square;
      betakt.col(j) = rmvnorm_col(meanbeta, covbeta);
      //Akt.col(j) = meanA;
    }
    beta.slice(k-1) = betakt.t();
  }
  return beta;
}


double dmvnrm_arma(arma::rowvec x,
                   arma::rowvec mean,
                   double sigma_square,
                   bool logd = false) {

  double out = dot(x-mean, x-mean);
  out = out/(-2 * sigma_square);
  if (logd == false) {
    out = exp(out);
  }
  return(out);
}

double dmvnrm_arma_matrix(arma::mat x,
                          arma::mat mean,
                          double sigma_square,
                          bool logd = false) {
  int n = x.n_rows;
  int m = x.n_cols;
  x.reshape(1, n*m);
  mean.reshape(1, n*m);

  double ans = dmvnrm_arma(x, mean,sigma_square, logd);
  return ans;
}


double compute_likelihood(arma::rowvec eta, arma::mat mu, arma::mat U, double sigma_square){
  int N = U.n_rows;
  int q = U.n_cols;
  double likeli = 0;
  double fij;
  double sd = std::sqrt(sigma_square);
  mu = mu/sd;
  eta = eta/sd;
  for(int i=0; i<N; i++){
    for(int j=0; j<q; j++){
      fij = fastncdf( (mu(i,j) - eta(U(i,j)))) -
        fastncdf( (mu(i,j) - eta(U(i,j)+1) ));
      if(fij==0){
        likeli -= 15;
      }else{
        likeli += log(fij);
      }
    }
  }
  mu = mu*sd;
  eta = eta*sd;
  return (likeli);
}

arma::mat compute_betaZ( arma::mat& betak,
                      arma::mat& datak,
                      int Q, int S){
  int NN = datak.n_rows;
  arma::mat Zk(NN, S);
  arma::mat betaZ(NN,Q);

  arma::mat tmp(Q,1);
  Zk = colsome(datak, n_to_m(2, 2+S-1));

  //AZ = join_rows(repp(1,NN).t(),Zk) * Ak.t();
  betaZ = join_rows(repp(1,NN).t(), Zk) * betak.t();

  return betaZ;
}

arma::colvec compute_BX_j2(arma::mat& alphakt, arma::mat& gammakt,
                               arma::mat& Rk, arma::mat& betak,
                               arma::mat& Zkt, arma::mat& datak,  int j,
                               int D, int Q, int S, int B){
  int NN = datak.n_rows;
  arma::rowvec ind(D);
  ind = n_to_m(j*D, j*D+D-1);
  arma::mat tmp(1,1);

  arma::mat Zk(NN, S);
  arma::mat bk(NN, B);

  arma::colvec BX(NN);
  arma::mat zalpha(D, NN);
  arma::mat bxbi(D, NN);
  arma::colvec BB(D);
  arma::colvec BBR(D);

  Zk = colsome(datak, n_to_m(2, 2+S-1));
  bk = colsome(datak, n_to_m(2+S, 1+S+B));

  zalpha =  rowsome(alphakt, ind) * Zk.t();
  bxbi =  rowsome(gammakt,ind) * bk.t();
  //zalpha.print();

  for(int i=0; i<NN; i++){
    BB = zalpha.col(i) + bxbi.col(i);
    BBR = BB % Rk.row(j).t();
    tmp = BBR.t() * Zkt.col(i);
    BX(i) = tmp(0,0);

  }
  return BX;
}


arma::cube update_R(int H, arma::rowvec& e,
                             arma::cube& alpha, arma::cube& gamma,
                             arma::cube R, arma::cube& beta,
                             arma::mat& Z, arma::mat& data,
                             int D, int Q, int S, int B,
                             arma::rowvec& a, arma::mat& U,
                             arma::mat mu, arma::mat Y,
                             double rho, double sigma_square){
  arma::cube R_new = R;
  arma::cube Likeli = R;
  arma::mat Rk0(Q,D);

  int NN = data.n_rows;
  arma::rowvec pt_id(NN);
  arma::rowvec tmp_ind(NN);
  arma::mat alphakt(D*Q, S);
  arma::mat gammakt(Q*D,B);
  arma::mat Rk(Q,D);
  arma::mat betak(Q, 1+S);
  arma::mat Yk(NN, Q);
  arma::mat Uk(NN, Q);
  arma::mat datak(NN, 2+S+B);
  arma::mat Zk(NN, D);
  arma::mat muk(NN, Q);
  arma::mat muk1(NN,Q);
  arma::mat muk0(NN,Q);
  arma::rowvec PP(2);

  int ll, LL;
  for(int k=1; k<=H; k++){
    ll = getind(e, k).n_cols;
    pt_id.reshape(1,ll);
    pt_id = colsome(unique(data.col(0).t()), getind(e,k));

    LL = whichin(data.col(0).t(), pt_id).n_cols;
    tmp_ind.reshape(1,LL);
    tmp_ind = whichin(data.col(0).t(), pt_id);

    Yk.reshape(LL, Q);
    datak.reshape(LL, 2+S+B);
    Zk.reshape(LL, D);
    muk.reshape(LL,Q);
    muk1.reshape(LL,1);
    muk0.reshape(LL,1);
    Uk.reshape(LL,Q);

    arma::mat Zkt(D, LL);

    Yk = rowsome(Y, tmp_ind);
    datak = rowsome(data, tmp_ind);
    Zk = rowsome(Z, tmp_ind);
    muk = rowsome(mu, tmp_ind);
    Uk = rowsome(U, tmp_ind);
    Zkt = Zk.t();

    arma::mat betaZ(LL, Q);
    arma::mat BX(LL, 1);
    alphakt = alpha.slice(k-1).t();
    gammakt = gamma.slice(k-1).t();
    Rk = R.slice(k-1);
    betak = beta.slice(k-1);

    betaZ = compute_betaZ(betak, datak, Q, S);

    for(int i=0; i<Q; i++){
      for(int j=0; j<D; j++){
        Rk0 = Rk;
        PP.fill(0);

        if(Rk(i,j)==1){
          PP(0) = log(rho) + compute_likelihood(a,  muk.col(i), Uk.col(i), sigma_square);
          muk1 = muk.col(i);
          Rk0(i,j) = 0;
          BX = compute_BX_j2(alphakt, gammakt, Rk0, betak, Zkt, datak, i, D, Q, S,  B);

          muk0 = betaZ.col(i) + BX;
          PP(1) = log(1-rho) +  compute_likelihood(a,  muk0, Uk.col(i), sigma_square);
        }else{
          PP(1) = log(1-rho) +  compute_likelihood(a,  muk.col(i), Uk.col(i), sigma_square);
          muk0 = muk.col(i);
          Rk0(i,j) = 1;
          BX = compute_BX_j2(alphakt, gammakt, Rk0, betak, Zkt, datak, i, D, Q, S, B);
          muk1 = betaZ.col(i) + BX;

          PP(0) = log(rho) +  compute_likelihood(a,  muk1, Uk.col(i), sigma_square);
        }

        PP = PP-max(PP);
        Likeli(i,j,k-1) = exp(PP[0]) / (exp(PP[0]) + exp(PP[1]));
        R_new(i,j,k-1) = Rcpp::rbinom(1,1,Likeli(i,j,k-1))(0);
        Rk(i,j) = R_new(i,j,k-1);

        if(R_new(i,j,k-1)==0){
          muk.col(i) = muk0;
        }else{
          muk.col(i) = muk1;
        }
      }
    }
  }

  return R_new;
}

List update_e(int H, arma::rowvec e,
                    arma::cube alpha, arma::cube gamma,
                    arma::cube R, arma::cube beta,
                    arma::mat& Z, arma::mat& data,
                    int D, int Q, int S, int B,
                    arma::rowvec& a, arma::mat& U,
                    arma::mat mu, arma::mat Y,
                    int N, double sigma_square,
                    double m0, double rho, int H_max){
  int e0,ll;
  List ans;
  arma::rowvec Ns(N);
  arma::rowvec PP(N);
  arma::rowvec tmp_ind(N);
  arma::mat Yk(N, Q);
  arma::mat Uk(N,Q);
  arma::mat datak(N, 2+S+B);
  arma::mat Zkt(D,N);
  arma::mat muk0(N,Q);
  arma::mat muk1(N,Q);
  arma::cube alpha_new(S,Q*D, H);
  arma::cube gamma_new(B,D*Q, H);
  arma::cube beta_new(Q,1+S, H);
  arma::cube R_new(Q, D, H);

  arma::mat alphakt(D*Q, S);
  arma::mat gammakt(D*Q,B);
  arma::mat Rk(Q,D);
  arma::mat betak(Q,1+S);

  for(int i=0; i<N; i++){
    e0 = e(i);

    if(getind(e, e0).n_cols==1){
      alpha.shed_slice(e0-1);
      gamma.shed_slice(e0-1);
      R.shed_slice(e0-1);
      beta.shed_slice(e0-1);

      for(int kk=0; kk<N; kk++){
        if(e(kk)>e0){
          e(kk) = e(kk)- 1;
        }
      }
      H --;
    }
    Ns.reshape(1, H);
    Ns.fill(0);
    for(int j=0; j<N; j++){
      if(j==i)
        continue;

      Ns(e(j)-1) ++;
    }
    PP.reshape(1, H+1);
    PP.fill(0);

    ll = getind(data.col(0).t(), i+1).n_cols;
    tmp_ind.reshape(1,ll);
    tmp_ind = getind(data.col(0).t(), i+1);

    Yk.reshape(ll, Q);
    Uk.reshape(ll, Q);
    datak.reshape(ll, 2+B+S);
    Zkt.reshape(D,ll);
    muk0.reshape(ll,Q);
    muk1.reshape(ll,Q);

    Yk = rowsome(Y, tmp_ind);
    Uk = rowsome(U, tmp_ind);
    datak = rowsome(data, tmp_ind);
    Zkt = rowsome(Z, tmp_ind).t();

    for(int k=0; k<H; k++){
      alphakt = alpha.slice(k).t();
      gammakt = gamma.slice(k).t();
      Rk = R.slice(k);
      betak = beta.slice(k);
      muk0 = compute_muk2( alphakt, gammakt,
                               Rk, betak, Zkt, datak,D,Q,S,B);
      PP(k) = std::log(Ns(k)) + compute_likelihood(a,  muk0, Uk, sigma_square);
    }

    if(H < H_max){
      for(int j=0; j<Q*D; j++){

        for(int l=0; l<S; l++){
          alphakt(j,l) = R::rnorm(0,0.5);
        }
        for(int l=0; l<B; l++){
          gammakt(j,l) = R::rnorm(0,0.5);
        }
      }

      for(int j=0; j<Q; j++){
        for(int l=0; l<D; l++){
          Rk(j,l) = R::rbinom(1,rho);
        }
        for(int l=0; l<1+S; l++){
          betak(j,l) = R::rnorm(0,0.5);
        }
      }

      muk1 = compute_muk2(alphakt, gammakt,
                              Rk, betak, Zkt, datak,D,Q,S,B);

      PP(H) = std::log(m0) + compute_likelihood(a,  muk1, Uk, sigma_square);
      PP = PP-max(PP);
      PP = exp(PP);
      PP = PP/sum(PP);
    }else{
      PP(H) = PP(0);
      PP = PP-max(PP);
      PP = exp(PP);
      PP(H) = 0;
      PP = PP/sum(PP);
    }


    e(i) =  rmunoim(PP);

    if(e(i)==(H+1)){

      alpha_new.reshape(S,Q*D, H+1);
      gamma_new.reshape(B,Q*D, H+1);
      R_new.reshape(Q,D, H+1);
      beta_new.reshape(Q,1+S, H+1);

      alpha_new.slices(0,H-1) = alpha;
      gamma_new.slices(0,H-1) = gamma;
      R_new.slices(0,H-1) = R;
      beta_new.slices(0,H-1) = beta;

      alpha_new.slice(H) = alphakt.t();
      gamma_new.slice(H) = gammakt.t();
      R_new.slice(H) = Rk;
      beta_new.slice(H) = betak;


      alpha.reshape(S,Q*D, H+1);
      gamma.reshape(B,Q*D, H+1);
      R.reshape(Q,D, H+1);
      beta.reshape(Q,1+S, H+1);


      alpha= alpha_new;
      gamma = gamma_new;
      R = R_new;
      beta=beta_new;

      H++;
    }
  }

  ans["alpha"] = alpha;
  ans["gamma"] = gamma;
  ans["R"] = R;
  ans["beta"] = beta;
  ans["H"] = H;
  ans["e"] = e;
  return ans;
}


double update_sig_square(arma::mat& Err, double a, double b){
  double s = accu(Err%Err);
  int n = Err.n_elem;
  double ans = 1/R::rgamma(a+n/2, 1/(1/b+s/2));
  return ans;
}

double update_sig_square_cub(arma::cube& Y, double a, double b){
  double s = accu(Y%Y);
  int n = Y.n_elem;
  double ans = 1/R::rgamma(a+n/2, 1/(1/b+s/2));
  return ans;
}

arma::mat update_omega(arma::mat& Y, arma::mat& mu, arma::mat& Comega, double sigma_square, int Q){
  arma::mat II(Q,Q);
  int n = Y.n_rows;
  II.eye();
  arma::mat Sigma(Q,Q);
  Sigma = inv(Comega.i() + II/sigma_square);
  arma::mat mean(n, Q);
  arma::mat omega(Q, n);
  mean = (Y-mu) * Sigma/sigma_square;
  arma::mat meant = mean.t();
  for(int i=0; i<n; i++){
    omega.col(i) = rmvnorm_col(meant.col(i), Sigma);
  }
  return omega.t();
}

double update_gamma_sigma_square(arma::cube& gamma, int B, double a, double b){
  int n = gamma.n_slices * gamma.n_cols;
  arma::mat tmp(B, n);
  for(int i=0; i<gamma.n_slices; i++){
    tmp.cols(i*gamma.n_cols, (i+1)*gamma.n_cols-1) = gamma.slice(i);
  }
  double s=0;
  arma::mat KK(B, B);
  arma::mat tt(1,1);
  KK = makeP(B);
  for(int i=0; i<n; i++){
    tt = tmp.col(i).t() * KK * tmp.col(i);
    s += tt(0,0);
  }
  double ans = 1/R::rgamma(a+n*(KK.n_cols-2)/2, 1/(1/b+s/2));
  //cout<<a+n*(KK.n_cols-2)/2<<" and "<<1/(1/b+s/2)<<endl;
  return ans;

}


arma::mat update_Y(arma::mat& U, arma::rowvec& a, arma::mat mu, int Q, double sigma_square){
  int n = mu.n_rows;
  arma::mat Y(n, Q);
  double l = 8*sqrt(sigma_square);
  for(int i=0; i<n; i++){
    for(int j=0; j<Q; j++){
      if(mu(i,j)<a(U(i,j)) - l){
        Y(i,j) = a(U(i,j));
      }else if(mu(i,j)>a(U(i,j)+1) + l){
        Y(i,j) = a(U(i,j)+1);
      }else{
        Y(i,j) = r_truncnorm(mu(i,j), sqrt(sigma_square), a(U(i,j)) , a(U(i,j)+1));
      }
    }
  }
  return Y;
}

arma::rowvec update_a(arma::mat& U, arma::mat mu, arma::rowvec a, double sigma,
                            double step){
  int n = a.n_cols;
  int n1 = U.n_rows;
  int n2 = U.n_cols;
  double t1, t2;
  double likelinow,likelinew, a_new;
  for(int i=2; i<n-1; i++){
    a_new = a(i) + R::rnorm(0, step);
    likelinew = 0;
    likelinow = 0;
    for(int j=0; j<n1;j++){
      for(int k=0; k<n2; k++){
        if(U(j,k)==(i-1)){
          t1 = fastncdf((a_new-mu(j,k))/sigma) - fastncdf((a(i-1)-mu(j,k))/sigma);
          t2 = fastncdf((a(i) - mu(j,k))/sigma) - fastncdf((a(i-1)-mu(j,k))/sigma);
          if(t1*t2!=0){
            likelinew += log(t1);
            likelinow += log(t2);
          }

        }
        if(U(j,k)==i){
          t1 = fastncdf((a(i+1) - mu(j,k))/sigma) - fastncdf((a_new-mu(j,k))/sigma);
          t2 = fastncdf((a(i+1) - mu(j,k))/sigma) - fastncdf((a(i)-mu(j,k))/sigma);
          if(t1*t2!=0){
            likelinew += log(t1);
            likelinow += log(t2);
          }
        }
      }

    }
    if(log(R::runif(0,1))< (likelinew-likelinow))
      a(i) = a_new;
  }

  return a;
}


arma::vec Mahalanobis(arma::mat x, arma::rowvec center, arma::mat cov) {
  int n = x.n_rows;
  arma::mat x_cen;
  x_cen.copy_size(x);
  for (int i=0; i < n; i++) {
    x_cen.row(i) = x.row(i) - center;
  }
  return sum((x_cen * cov.i()) % x_cen, 1);
}

arma::vec dmvnorm_arma(arma::mat x, arma::rowvec mean, arma::mat sigma, bool log = false) {
  arma::vec distval = Mahalanobis(x,  mean, sigma);
  double logdet = sum(arma::log(arma::eig_sym(sigma)));
  arma::vec logretval = -( (x.n_cols * log2pi + logdet + distval)/2  ) ;

  if (log) {
    return(logretval);
  } else {
    return(exp(logretval));
  }
}

arma::mat update_Comega(arma::mat& omega, arma::mat Comega, int Q, double sigma_square,
                         double step, double lower, double upper){
  double low1, low2, up1, up2,  prior, rho_new, likelinow, likelinew;
  arma::mat Comega_now = Comega;
  arma::mat Comega_new = Comega;
  arma::rowvec mean(Q);
  mean.fill(0);

  likelinow = sum(dmvnorm_arma(omega,mean, Comega_now*sigma_square, true));

  for(int i=0; i<Q-1; i++){
    for(int j=i+1; j<Q; j++){
      low1 = max(Comega_now(i,j)-step, lower);
      up1 = min(Comega_now(i,j)+step, upper);
      rho_new = R::runif(low1, up1);
      Comega_new(i,j) = rho_new;
      Comega_new(j,i) = rho_new;
      likelinew = sum(dmvnorm_arma(omega,mean, Comega_new*sigma_square, true));
      low2 = max(rho_new-step, lower);
      up2 = min(rho_new+step, upper);

      prior = log((up1-low1)/(up2-low2));

      if(log(R::runif(0,1))<likelinew-likelinow + prior){
        likelinow = likelinew;
        Comega_now(i,j) = rho_new;
        Comega_now(j,i) = rho_new;
      }else{
        Comega_new(i,j) = Comega_now(i,j);
        Comega_new(j,i) = Comega_now(i,j);
      }
    }
  }
  return (Comega_now);
}



