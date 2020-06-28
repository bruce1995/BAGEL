## usethis namespace: start
#' @importFrom Rcpp sourceCpp
## usethis namespace: end
NULL

update_rho <- function(R, alpha=1, beta=10){
  NR <- prod(dim(R))
  NX <- sum(R)
  rbeta(1,alpha+NX, beta+NR-NX)
}

#' @name main_mcmc
#' @aliases main_mcmc
#' @title The function returns the posted-brun-in MCMC samples thinned by the thinning factor
#' @description This function returns the posted-brun-in MCMC samples thinned by the thinning factor, provided observed data and initial values for some parameters.
#' @param Data {a list giving observed data. It includes:
#' \code{U} giving the depression scores, each row corresponds to one visit;
#' \code{Z} giving the binary indicator matrix for drug usage information, each row corresponds to one visit;
#' \code{X} giving the covariates matrix, it also includes the patient ID (1st column) and visit time (2nd column);
#' \code{D} giving the number of drugs;
#' \code{Q} giving the number of depression scores;
#' \code{S} giving the number of covariates;
#' \code{N} giving the number of patients in the dataset;
#' \code{B} giving the number of bases for P-spline.}
#' @param Burnin number of burn-in iterations in MCMC
#' @param Niter number of iterations after burn-in in MCMC
#' @param thin the MCMC thinning factor
#' @param hyper_parameter a list giving hyperparameters used in MCMC. See details below for more information.
#' @param seed the starting number used to generate random numbers
#' @return {the posted-brun-in MCMC samples thinned by the thinning factor, along with the following:
#' \code{D} the number of drugs;
#' \code{Q} the number of depression scores;
#' \code{S} the number of covariates;
#' \code{knots}  the knots for P-spline;
#' \code{B} the number of bases for P-spline;
#' \code{sigma_square} the variance of noise for the latent continuous depression scores;
#' \code{m0} the concentration parameter for Dirichlet Process.
#' }
#' @details
#' \code{hyper_parameter} is a list giving hyperparameters used in MCMC. It includes:
#' \code{Comega_init} giving the intial value of \code{Comega}, which is the correlation matrix which captures the dependencies among depression items;
#' \code{rho_init} giving the intial value of \code{rho}, which is the prior probability of the binary indicator cube entries being 1;
#' \code{H_init} giving the intial value of \code{H}, which is the number of clusters;
#' \code{a_init} giving the intial value of \code{a}, which is the thresholds that connects the latent and observed depression scores;
#' \code{e_init} giving the intial value of \code{e}, which is the clustering membership vector;
#' \code{alpha_init} giving the intial value of \code{alpha}, which is the linear coefficients dependent on drug effect (it should be a cube, each slice corresponds to one cluster, each slice should be a 'S' by 'DQ' matrix);
#' \code{gamma_init} giving the intial value of \code{gamma}, which is the linear coefficients for the P-spline (it should be a cube, each slice corresponds to one cluster, each slice should be a 'B' by 'DQ' matrix);
#' \code{beta_init} giving the intial value of \code{beta}, which is the linear coefficients not dependent on drug effect (it should be a cube, each slice corresponds to one cluster, each slice should be a 'Q' by '1+S' matrix with the first column being the intercept);
#' \code{R_init} giving the intial value of \code{R}, which is the binary indicator cube (it should be a cube, each slice corresponds to one cluster, each slice should be a 'Q' by 'D' matrix);
#' \code{sigma_square_alpha_init} giving the intial value of \code{sigma_square_alpha}, which is the variance of \code{alpha};
#' \code{sigma_square_beta_init} giving the intial value of \code{sigma_square_beta}, which is the variance of \code{beta};
#' \code{sigma_square_gamma_init} giving the intial value of \code{sigma_square_gamma}, which is the variance of \code{gamma};
#' \code{sigma_square} giving the variance of noise for the latent continuous depression scores;
#' \code{m0} giving the concentration parameter for Dirichlet Process;
#' \code{H_max} giving the maximum number of clusters allowed;
#' \code{a_rho} giving the first hyper-parameter of \code{rho} with default value being 1, where we assume \code{rho}~Beta(alpha, beta);
#' \code{b_rho} giving the second hyper-parameter of \code{rho};
#' \code{step_a} giving the step size for Metropolis–Hastings algorithm updating \code{a};
#' \code{step_Comega} giving the step size for Metropolis–Hastings algorithm updating \code{Comega};
#' \code{a_beta} giving the first hyper-parameter of \code{sigma_square_beta}, where we assume variance of \code{beta} is sampled from inv-Gamma(a,b);
#' \code{b_beta} giving the second hyper-parameter of \code{sigma_square_beta};
#' \code{a_alpha} giving the first hyper-parameter of \code{sigma_square_alpha}, where we assume variance of \code{alpha} is sampled from inv-Gamma(a,b);
#' \code{b_alpha} giving the second hyper-parameter of \code{sigma_square_alpha};
#' \code{a_gamma} giving the first hyper-parameter of \code{sigma_square_gamma}, where we assume variance of \code{gamma} is sampled from inv-Gamma(a,b);
#' \code{b_gamma} giving the second hyper-parameter of \code{sigma_square_gamma}.
#' @examples
#' \dontrun{
#' library(BAGEL)
#' data("simulated_data")
#' mcmc <- main_mcmc(Data=simulated_data, 2,10,5)
#' x_new <- simulated_data$X[10,]
#' z_new <- simulated_data$Z[10,]
#' depression_prob_predict(mcmc, z_new, x_new)
#' }
#' @export
main_mcmc <- function(Data, Burnin, Niter, thin = 1, hyper_parameter = list(), seed = 1){
  require(splines)
  require(mvtnorm)
  set.seed(seed)

  #########################################################################################
  # hyperparameter information
  #########################################################################################
  U <- Data$U; Z <- Data$Z; X <- Data$X; D <- Data$D; Q <- Data$Q; S <- Data$S; N <- Data$N
  NN <- dim(U)[1]
  B <- ifelse(is.null(Data$B),10,Data$B)
  if (B <= 4)
    stop("Since we use cubic spline, B must be larger than 4")

  H_init <- ifelse(is.null(hyper_parameter$H_init),2,hyper_parameter$H_init)
  if(is.null(hyper_parameter$prob_init)){
    prob_init <- rep(1/H_init, H_init)
  }else{
    prob_init <- hyper_parameter$prob_init
  }
  if(is.null(hyper_parameter$Comega_init)){
    Comega_init <- diag(Q)
  }else{
    Comega_init <- hyper_parameter$Comega_init
  }
  if(is.null(hyper_parameter$a_init)){
    a_init <- c(-100,0,10,20,100)
  }else{
    a_init <- hyper_parameter$a_init
  }
  if (length(prob_init) != H_init)
    stop("The length of initial cluster membership probability 'prob_init' must equal to the initial number of clusters 'H_init'.")
  if (dim(Comega_init)[1] != Q)
    stop("The dim of initial correlation matrix 'Comega_init' must equal to the number of depression items 'Q'.")

  e_init <- sample.int(H_init, size = N, replace = T, prob = prob_init)
  rho_init <- ifelse(is.null(hyper_parameter$rho_init),0.1,hyper_parameter$rho_init)
  sigma_square_alpha_init <- ifelse(is.null(hyper_parameter$sigma_square_alpha_init),1,hyper_parameter$sigma_square_alpha_init)
  sigma_square_beta_init <- ifelse(is.null(hyper_parameter$sigma_square_beta_init),1,hyper_parameter$sigma_square_beta_init)
  sigma_square_gamma_init <- ifelse(is.null(hyper_parameter$sigma_square_gamma_init),1,hyper_parameter$sigma_square_gamma_init)
  sigma_square <- ifelse(is.null(hyper_parameter$sigma_square),1,hyper_parameter$sigma_square)

  m0 <- ifelse(is.null(hyper_parameter$m0),1,hyper_parameter$m0)
  H_max <- ifelse(is.null(hyper_parameter$H_max),10,hyper_parameter$H_max)
  a_rho <- ifelse(is.null(hyper_parameter$a_rho),1,hyper_parameter$a_rho)
  b_rho <- ifelse(is.null(hyper_parameter$b_rho),10,hyper_parameter$b_rho)

  step_a <- ifelse(is.null(hyper_parameter$step_a),0.3,hyper_parameter$step_a)
  step_a <- ifelse(is.null(hyper_parameter$step_a),0.3,hyper_parameter$step_a)
  step_Comega <- ifelse(is.null(hyper_parameter$step_Comega),100,hyper_parameter$step_Comega)
  a_beta <- ifelse(is.null(hyper_parameter$a_beta),3,hyper_parameter$a_beta)
  b_beta <- ifelse(is.null(hyper_parameter$b_beta),10,hyper_parameter$b_beta)
  a_alpha <- ifelse(is.null(hyper_parameter$a_alpha),3,hyper_parameter$a_alpha)
  b_alpha <- ifelse(is.null(hyper_parameter$b_alpha),10,hyper_parameter$b_alpha)
  a_gamma <- ifelse(is.null(hyper_parameter$a_gamma),10,hyper_parameter$a_gamma)
  b_gamma <- ifelse(is.null(hyper_parameter$b_gamma),1,hyper_parameter$b_gamma)

  alpha_init  <- array(rnorm(S*Q*D*H_init,sd=1), dim=c(S,D*Q, H_init))
  gamma_init  <- array(rnorm(B*Q*D*H_init,sd=1), dim=c(B,Q*D, H_init))
  beta_init   <- array(rnorm(Q*(1+S)*H_init,sd=1), dim=c(Q,1+S, H_init))
  R_init      <- array(rbinom(D*Q*H_init, size = 1, prob = rho_init), dim=c(Q,D, H_init))


  #########################################################################################
  # generate the P spline base
  #########################################################################################
  tt <- X[,2]/max(X[,2])
  knots <- c(rep(min(tt),4)-c(0.2,0.15,0.1,0.05),
             quantile(tt,(1:(B-4))/(B-3)),
             rep(max(tt), 4) + c(0.05,0.1,0.15,0.2))
  pbase <- splineDesign(knots, tt, outer.ok = T)
  data  <- cbind(X, pbase)
  #########################################################################################
  # preprocess
  #########################################################################################
  mcmc <- NULL
  mcmc$e      <- matrix(NA, nrow = Niter %/% thin, ncol = N)
  mcmc$alpha  <- array(NA, dim=c(S,D*Q, H_max, Niter %/% thin))
  mcmc$gamma  <- array(NA, dim=c(B,Q*D, H_max, Niter %/% thin))
  mcmc$beta   <- array(NA, dim=c(Q,1+S, H_max, Niter %/% thin))
  mcmc$R      <- array(NA, dim=c(Q,D, H_max, Niter %/% thin))
  mcmc$H      <- rep(NA, Niter %/% thin)
  mcmc$a      <- matrix(NA, nrow = Niter %/% thin, ncol = length(a_init))
  mcmc$Comega <- array(NA, dim = c(Q,Q,Niter %/% thin))

  sigma_square_alpha <- sigma_square_alpha_init
  sigma_square_beta <- sigma_square_beta_init
  sigma_square_gamma <- sigma_square_gamma_init

  H <- H_init; a <- a_init; rho <- rho_init; Comega <- Comega_init
  alpha <- alpha_init; gamma <- gamma_init; beta <- beta_init; R <- R_init; e <- e_init
  omega <- rmvnorm(NN, rep(0,Q), Comega_init)
  mu    <- compute_mu(H, e,  alpha, gamma, R, beta, Z, data, D, Q, S, B)
  Y     <- update_Y(U, a, mu+omega, Q,sigma_square )

  for(iter in 1:Burnin){
    RETURNALPHA <- update_alpha(H, e, alpha, gamma, R, beta, Z, data, D, Q, S, B,
                                mu, Y-omega, sigma_square, sigma_square_alpha)
    alpha <- RETURNALPHA$alpha
    mu    <- RETURNALPHA$mu

    RETURNGAMMA <- update_gamma(H, e, alpha, gamma, R, beta, Z, data, D, Q, S, B,
                                mu, Y-omega, sigma_square, sigma_square_gamma)
    gamma <- RETURNGAMMA$gamma
    mu    <- RETURNGAMMA$mu

    beta <- update_beta(H, e, alpha, gamma, R, beta, Z,
                        data, D, Q, S, B, Y-omega, sigma_square, sigma_square_beta)
    mu   <- compute_mu(H, e,alpha, gamma, R, beta, Z, data, D, Q, S, B)

    R  <- update_R(H, e, alpha, gamma, R, beta, Z, data, D, Q, S, B,
                   a, U, mu,Y-omega, rho ,sigma_square )
    mu <- compute_mu(H, e,  alpha, gamma,  R, beta, Z, data, D, Q, S, B)

    RETURNE <- update_e(H, e, alpha, gamma, R, beta,
                        Z, data, D, Q, S, B,a, U, mu, Y-omega, N, sigma_square, m0, rho, H_max)
    H     <- RETURNE$H
    e     <- RETURNE$e
    alpha <- RETURNE$alpha
    gamma <- RETURNE$gamma
    R     <- RETURNE$R
    beta  <- RETURNE$beta
    mu    <- compute_mu(H, e, alpha, gamma, R, beta, Z, data, D, Q, S, B)

    rho                <- update_rho(R, a_rho, b_rho)
    Y                  <- update_Y(U, a, mu+omega, Q,sigma_square )
    a                  <- update_a(U, mu+omega, a, sqrt(sigma_square),step_a)
    omega              <- update_omega(Y, mu, Comega, sigma_square, Q)
    Comega             <- update_Comega(omega, Comega, Q, sigma_square, step = step_Comega/NN)
    sigma_square_beta  <- update_sig_square_cub(beta, a=a_beta, b=b_beta)
    sigma_square_alpha <- update_sig_square_cub(alpha, a=a_alpha, b=b_alpha)
    sigma_square_gamma <- update_gamma_sigma_square(gamma, B, a=a_gamma, b=b_gamma)
  }
  for(iter in 1:Niter){
    RETURNALPHA <- update_alpha(H, e, alpha, gamma, R, beta, Z, data, D, Q, S, B,
                                mu, Y-omega, sigma_square, sigma_square_alpha)
    alpha <- RETURNALPHA$alpha
    mu    <- RETURNALPHA$mu

    RETURNGAMMA <- update_gamma(H, e, alpha, gamma, R, beta, Z, data, D, Q, S, B,
                                mu, Y-omega, sigma_square, sigma_square_gamma)
    gamma <- RETURNGAMMA$gamma
    mu    <- RETURNGAMMA$mu

    beta <- update_beta(H, e, alpha, gamma, R, beta, Z,
                        data, D, Q, S, B, Y-omega, sigma_square, sigma_square_beta)
    mu   <- compute_mu(H, e,alpha, gamma, R, beta, Z, data, D, Q, S, B)

    R  <- update_R(H, e, alpha, gamma, R, beta, Z, data, D, Q, S, B,
                   a, U, mu,Y-omega, rho ,sigma_square )
    mu <- compute_mu(H, e,  alpha, gamma,  R, beta, Z, data, D, Q, S, B)

    RETURNE <- update_e(H, e, alpha, gamma, R, beta,
                        Z, data, D, Q, S, B,a, U, mu, Y-omega, N, sigma_square, m0, rho, H_max)
    H     <- RETURNE$H
    e     <- RETURNE$e
    alpha <- RETURNE$alpha
    gamma <- RETURNE$gamma
    R     <- RETURNE$R
    beta  <- RETURNE$beta
    mu    <- compute_mu(H, e, alpha, gamma, R, beta, Z, data, D, Q, S, B)

    rho                <- update_rho(R)
    Y                  <- update_Y(U, a, mu+omega, Q,sigma_square )
    a                  <- update_a(U, mu+omega, a, sqrt(sigma_square),step_a)
    omega              <- update_omega(Y, mu, Comega, sigma_square, Q)
    Comega             <- update_Comega(omega, Comega, Q, sigma_square, step = step_Comega/NN)
    sigma_square_beta  <- update_sig_square_cub(beta, a=a_beta, b=b_beta)
    sigma_square_alpha <- update_sig_square_cub(alpha, a=a_alpha, b=b_alpha)
    sigma_square_gamma <- update_gamma_sigma_square(gamma, B, a=a_gamma, b=b_gamma)
    if(iter%%thin==0){
      iter_record <- iter/thin
      mcmc$e[iter_record,]   <- e
      mcmc$alpha[,,1:H, iter_record] <- alpha
      mcmc$gamma[,,1:H, iter_record] <- gamma
      mcmc$beta[,,1:H, iter_record]  <- beta
      mcmc$R[,,1:H, iter_record]     <- R
      mcmc$H[iter_record] <- H
      mcmc$a[iter_record,] <-a
      mcmc$Comega[,,iter_record] <- Comega
    }
  }
  mcmc$D <- D; mcmc$Q <- Q; mcmc$S <- S; mcmc$B <- B; mcmc$knots <- knots; mcmc$sigma_square <- sigma_square; mcmc$m0 <- m0
  return (mcmc)
}

#' @name depression_prob_predict
#' @aliases depression_prob_predict
#' @title The prediction function
#' @description This function returns the predicted probability of reporting depression for each depression item for a new visit, provided covariates, ART usage information, and previous visit time
#' @param mcmc the MCMC output given by \link{main_mcmc}
#' @param z_new the binary indicator matrix for drug usage information for new visit to predict
#' @param x_new the covariates matrix for new visit to predict, it also includes the patient ID (1st column) and visit time (2nd column)
#' @param seed the starting number used to generate random numbers
#' @return the predicted probability of reporting depression for each depression item
#' @seealso
#' See \link{main_mcmc} for an example.
#' @export
depression_prob_predict <- function(mcmc, z_new, x_new,  seed = 1){
  set.seed(seed)
  D <- mcmc$D; Q <- mcmc$Q; S <- mcmc$S; B <- mcmc$B; sigma_square <- mcmc$sigma_square; m0 <- mcmc$m0; knots <- mcmc$knots

  Niter <- length(mcmc$H)
  Prob  <- matrix(0, nrow = Niter, ncol = Q)
  N <- dim(mcmc$e)[2]

  bx_new <- splineDesign(knots, x_new[2], outer.ok = T)
  datak <- c(x_new, bx_new)

  for(iter in 1:Niter){
    Zkt <- as.matrix(z_new)
    weights <- table(mcmc$e[iter,])/(N + m0)
    Comega <- mcmc$Comega[,,iter]
    Corr <- Comega + diag(Q)
    prob <- rep(0, Q)
    for(k in 1:length(weights)){
      alphakt <- t(mcmc$alpha[,,k,iter])
      gammakt <- t(mcmc$gamma[,,k,iter])
      Rk <- mcmc$R[,,k,iter]
      betak <- mcmc$beta[,,k,iter]
      muk <-compute_muk2(alphakt, gammakt, Rk, betak,
                         Zkt, t(as.matrix(datak)), D, Q, S, B)

      latent <- muk
      prob <- prob + weights[k] * pnorm(muk,sd=sqrt(sigma_square))
    }
    Prob[iter,] <- prob + 0.5 * m0 /(N+m0)
  }
  apply(Prob,2,mean)
}


#' @name generate_simulate_data
#' @aliases generate_simulate_data
#' @title The function generates simulated data
#' @description This function generates simulated data, provided the scale of the dataset
#' @param D the number of drugs
#' @param Q the number of depression scores
#' @param S the number of covariates
#' @param N the number of patients in the dataset
#' @param H the number of clusters
#' @param B the number of bases for P-spline
#' @param rho the prior probability of the binary indicator cube entries being 1
#' @param prob the probabilities of patients being assigned to clusters. If not given, it will be equal probabilities.
#' @param f the function designed to capture the longitudinally on depression. If not given, it will be constant 0
#' @param Comega the correlation matrix which captures the dependencies among depression items. If not given, it will be an identity matrix
#' @param a the thresholds that connects the latent and observed depression scores
#' @param sigma_square the variance of noise for the latent continuous depression scores
#' @param seed the starting number used to generate random numbers
#' @return a simulated dataset with observated data and unobserved parameters
#' @examples
#' D = 5; Q=3; S=5; N=100;H=10
#' simulated_data <- generate_simulate_data(D, Q, S, N, H)
#' @export
generate_simulate_data <- function(D, Q, S, N, H, B = 10, rho = 0.1, prob = NULL, f = NULL, Comega = NULL,
                                   a = c(-100,0,1,2,100), sigma_square = 1, seed = 1){
  set.seed(seed)
  require(splines)
  require(mvtnorm)
  if (B <= 4)
    stop("Since we use cubic spline, B must be larger than 4")
  simulated_data <- NULL
  simulated_data$D <- D
  simulated_data$Q <- Q
  simulated_data$S <- S
  simulated_data$N <- N
  simulated_data$H <- H
  simulated_data$B <- B
  simulated_data$a <- a

  simulated_data$beta   <- array(sample(c(0.5,1,1.5,-1.5,-1,-0.5), (1+S)*Q*H, replace = T), dim=c(Q,S+1, H))
  simulated_data$alpha   <- array(sample(c(0.5,1,1.5,-1.5,-1,-0.5), S*D*Q*H, replace = T), dim=c(S,D*Q, H))
  simulated_data$gamma  <- array(sample(c(1.5,1,0.5,-0.5,-1,-1.5), B*D*Q*H, replace = T), dim=c(B,D*Q, H))
  simulated_data$R <- array(rbinom(D*Q*H, size = 1, prob = rho), dim=c(Q,D,H))

  if(is.null(simulated_data$prob)){
    prob <- rep(1/H, H)
  }
  simulated_data$e <- sample.int(H, size = N, replace = T, prob = prob)

  if(is.null(f)){
    f <- function(x){0}
  }

  if(is.null(Comega)){
    simulated_data$Comega <- diag(Q)
  }else{
    simulated_data$Comega <- Comega
  }

  V <- rpois(N, 10)
  NN <- sum(V)
  WID <- visit <- c()
  simulated_data$Z <- matrix(rbinom(NN*D, size = 1, prob = 0.8), nrow = NN, ncol = D)
  for(i in 1:N){
    gap <- 1:V[i] + c(0, rnorm(V[i]-1, 0, 0.2))
    visit <- c(visit, gap-gap[1])
    WID <- c(WID, rep(i, V[i]))
  }
  Xtmp <- matrix(NA, nrow = NN, ncol = S)
  for(i in 1:(S%/%2)){
    Xtmp[,i] <- rbinom(NN, size = 1, prob = 0.6)
  }
  for(i in (1 + S%/%2):S){
    Xtmp[,i] <- rnorm(NN)
  }
  simulated_data$X <- X <- cbind(WID,visit/max(visit), Xtmp)

  tt <- X[,2]
  knots <- c(rep(min(tt),4)-c(0.2,0.15,0.1,0.05),
             quantile(tt,(1:(B-4))/(B-3)),
             rep(max(tt), 4) + c(0.05,0.1,0.15,0.2))
  pbase <- splineDesign(knots, tt, outer.ok = T)
  data  <- cbind(X, pbase)

  compute_mu0 <- function(H, e, alpha, gamma, R, beta, Z, data, D, Q, S, B){
    NN <- dim(data)[1]
    mu <- matrix(NA, nrow = NN, ncol = Q)

    for(k in 1:H){
      #loop for different cluster
      pt_id   <- which(e==k)
      tmp_ind <- which(data[,1]%in%pt_id)

      Xk <- data[tmp_ind,3:(2+S)]
      bx <- data[tmp_ind, 2]
      Zk <- Z[tmp_ind,]
      #get corresponding parameters
      alphak   <- alpha[,,k]
      gammak  <- gamma[,,k]
      Rk <- R[,,k]
      betak     <- beta[,,k]

      LL <- length(tmp_ind)
      BX <- matrix(NA, nrow = LL, ncol = Q)
      #the time-variant contribution (not influenced by drug)
      AZ <- cbind( rep(1,LL),Xk )%*%t(betak)

      #depends on drug
      zphi <- Xk %*% alphak

      #each row
      for(i in 1:LL){
        #compute each row that belongs to this cluster
        B  <- matrix(zphi[i,] + rep(f(bx[i]), D*Q), nrow = Q, ncol = D, byrow = TRUE)
        BR <- B*Rk
        BX[i,] <- BR %*% Zk[i,]
      }#end loop for i
      mu[tmp_ind,] <- AZ +BX
    }
    return (mu)
  }

  mu    <- compute_mu0(H, simulated_data$e,  simulated_data$alpha, simulated_data$gamma, simulated_data$R,
                       simulated_data$beta, simulated_data$Z, data, D, Q, S, B)

  U <- Y <- mu +  rmvnorm(NN, sigma = simulated_data$Comega, method = "chol") +
    matrix(rnorm(NN*Q, sd=sqrt(sigma_square)), nrow = NN, ncol = Q)

  for(i in 1:NN){
    for(j in 1:Q){
      U[i,j] = sum(Y[i,j]>simulated_data$a[-1])
    }
  }
  simulated_data$U <- U
  simulated_data
}
