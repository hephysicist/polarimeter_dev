
#include <chrono>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>

inline double linearjump(double t,double D, double a, double k1, double k2, double T)
{
  	if(t <=0)	return a+k1*(t-T/2.);
  	if(t >= T)	return a+D +k2*(t-T/2.);
  	return a+D*t/T + (k2*t*t - k1*(T-t)*(T-t))/2./T;
}

double fun (double *x, double *p) {
    return linearjump( x[0]- p[0],  p[1], p[2], p[3],p[4], p[5]);
}
#include <math.h>


double polarization(double P, double Pmax, double tau, double t) {
    //std::cout << "P = " << P << "  Pmax=" << Pmax << "  tau=" << tau << "  t= " << t << std::endl;
    //std::cout <<  (1.0 - exp(- t/tau)) << std::endl;
    return P + (Pmax-P) * (1.0 - exp(- t/tau));
}

double exp_jump(double time, double depol_time=0, double P0=0, double Pmax=-10, double tau=14, double DELTA=10, double T=1)
{
    if (time < depol_time - T/2) return polarization(P0, Pmax, tau, time);
    double P1 = polarization(P0, Pmax, tau, depol_time);
    double P2 = P1 + DELTA;
    if (time > depol_time + T/2) {
        return polarization(P2, Pmax, tau, time - depol_time);
    }
    double P1T = polarization(P0, Pmax, tau, depol_time - T/2);
    double P2T = polarization(P2, Pmax, tau, T/2);
    return P1T - (P1T - P2T) * (time - (depol_time - T/2)) / T;
}

TGraph gE;

class Fitter {
  std::ifstream file;
  public:
    Fitter(std::string file_name) : file(file_name) {
      if(!file) {
        std::cerr << "Unable to open file " << file_name << std::endl;
      }
      gP.Draw("a*");
    }
    TGraph gE;
    TGraphErrors gP;
    TGraphErrors gQ;
    TCanvas c;

    void loop(void) {
      char buf[65536];
      while(true) {
        while(!file.getline(buf,65536)) {
          gSystem->ProcessEvents();
          this_thread::sleep_for(100ms);
            //std::cout << buf << std::endl;
          file.clear();
        }
        //std::cout << buf << std::endl;
        std::istringstream iss(buf);
        time_t t;
        double P,dP,Q,dQ;
        double E,F;
        int n;

        iss >> n >> t >> F >> E>> P >> dP >>Q >> dQ;

        int i = gP.GetN();
        gP.SetPoint(i, t, P);
        gP.SetPointError(i, 0, dP);
        gQ.SetPoint(i, t, Q);
        gQ.SetPointError(i, 0, dQ);
        c.Modified();
        c.Update();
      }
    }
    void loop2(std::string filename, int T=10) {
      char buf[65536];
      while(true) {
        std::ifstream file2(filename);
        gP.Clear();
        gQ.Clear();
        while(file2.getline(buf,65536)) {
          std::istringstream iss(buf);
          time_t t;
          double P,dP,Q,dQ;
          double E,F;
          int n;

          iss >> n >> t >> F >> E>> P >> dP >>Q >> dQ;
          int i = gP.GetN();
        gP.SetPoint(i, t, E);
          gP.SetPoint(i, t, P);
          gP.SetPointError(i, 0, dP);
          gQ.SetPoint(i, t, Q);
          gQ.SetPointError(i, 0, dQ);
          c.Modified();
          c.Update();
        }
        auto tb = std::chrono::system_clock::now();
        while( std::chrono::system_clock::now() < tb + std::chrono::seconds(T)) {
          gSystem->ProcessEvents();
          this_thread::sleep_for(100ms);
        }
      }
    }

};


template <class F> double dgaus(F  func, double a, double b, double epsilon)
{
    const double Z1 = 1;
    const double HF = Z1/2;
    const double CST = 5*Z1/1000;

    double x[12] = { 0.96028985649753623,  0.79666647741362674,
        0.52553240991632899,  0.18343464249564980,
        0.98940093499164993,  0.94457502307323258,
        0.86563120238783174,  0.75540440835500303,
        0.61787624440264375,  0.45801677765722739,
        0.28160355077925891,  0.09501250983763744
    };

    double w[12] = { 0.10122853629037626,  0.22238103445337447,
        0.31370664587788729,  0.36268378337836198,
        0.02715245941175409,  0.06225352393864789,
        0.09515851168249278,  0.12462897125553387,
        0.14959598881657673,  0.16915651939500254,
        0.18260341504492359,  0.18945061045506850
    };

    double h, aconst, bb, aa, c1, c2, u, s8, s16, f1, f2;
    double xx[1];
    int i;

    //InitArgs(xx,params);

    h = 0;
    if (b == a) return h;
    aconst = CST/fabs(b-a);
    bb = a;
CASE1:
    aa = bb;
    bb = b;
CASE2:
    c1 = HF*(bb+aa);
    c2 = HF*(bb-aa);
    s8 = 0;
    for (i=0;i<4;i++) {
        u     = c2*x[i];
        xx[0] = c1+u;
        f1    = func(xx[0]);
        xx[0] = c1-u;
        f2    = func(xx[0]);
        s8   += w[i]*(f1 + f2);
    }
    s16 = 0;
    for (i=4;i<12;i++) {
        u     = c2*x[i];
        xx[0] = c1+u;
        f1    = func(xx[0]);
        xx[0] = c1-u;
        f2    = func(xx[0]);
        s16  += w[i]*(f1 + f2);
    }
    s16 = c2*s16;
    if (fabs(s16-c2*s8) <= epsilon*(1. + fabs(s16))) {
        h += s16;
        if(bb != b) goto CASE1;
    } else {
        bb = c1;
        if(1. + aconst*fabs(c2) != 1) goto CASE2;
        h = s8;  //this is a crude approximation (cernlib function returned 0 !)
    }
    return h;
}


class ExpErf {
    double Td;
    double delta; 
    double C;  
    double k1; 
    double k2; 
    double tau;  
    double power; 
    public:
    ExpErf(double td, double d, double c, double s1, double s2, double Tau, double Power) {
        Td = td;
        delta = d;
        C = c;
        k1 = s1;
        k2 = s2;
        tau = Tau;
        power = Power;
    };

    double operator() ( double x) {
        double t = x- Td;
        double hev = ( 1.0 - exp( - power * (1. + std::erf(t/tau)) )  ) / ( 1. - exp(-2.*power) );
        double result =  C + delta * hev + t*(k1 + (k2-k1)*hev);
        return result;
    };
};



//Многомерный интеграл

inline double experf(double *x, double *p)
{
    ExpErf fun(p[0], p[1], p[2], p[3],p[4], p[5],p[6] );
    double T= p[7];
    double result =  dgaus(fun,x[0]-T/2, x[0]+T/2, 1e-8)/T;
    return result;
}

class ExpJump {
    double Td;
    double P0;
    double Pmax;
    double tau;
    double delta;
    public:
    double TT;
    ExpJump(double depol_time=0, double P0_=0, double Pmax_=-10, double tau_=14, double DELTA_=10) {
        Td=depol_time;
        P0 = P0_;
        Pmax = Pmax_;
        tau = tau_;
        delta = DELTA_;
    }

    //double operator() (double t) {
    //    double result;
    //    if (t < Td) result = polarization(P0, Pmax, tau, t);
    //    double P1 = delta + polarization(P0, Pmax, tau, Td); //polarization just after the depolarization
    //    result = polarization(P1, Pmax, tau, t-Td);
    //    return result;
    //    //double sigma=10;
    //    //return result*1./sqrt(2.*M_PI)/sigma*exp(-0.5*pow(-(t-TT)/sigma,2));
    //};
    double operator() (double t) {
        double result;
        double f1 = polarization(P0, Pmax, tau, t);
        double P1 = delta + polarization(P0, Pmax, tau, Td); //polarization just after the depolarization
        double f2 = polarization(P1, Pmax, tau, t-Td);
        return f1 + (f2-f1)*(1.0 + erf((t-Td)/5))*0.5;
        //double sigma=10;
        //return result*1./sqrt(2.*M_PI)/sigma*exp(-0.5*pow(-(t-TT)/sigma,2));
    };

    static double root(double *x, double *p)
    {
        ExpJump fun(p[0], p[1], p[2], p[3], p[4]);
        fun.TT = x[0];
        double T= p[5];
        //double result =  dgaus(fun,x[0]-T/2, x[0]+T/2, 1e-8)/T;
        double result =  dgaus(fun,x[0], x[0]+T, 1e-8)/T;
        //double result =  dgaus(fun,x[0]-T+10, x[0]+10, 1e-8)/T;
        return result;
    }

    static void InitPars(TF1 * f, double td) {
        auto set = [&](int i, const char * name, double value) {
            f->SetParName(i,name);
            f->SetParameter(i,value);
        };
        set(0,"Td", td);
        f->SetParLimits(0,0, 1000000);
        set(1, "P0", -0.9);
        //f->SetParLimits(1,-0.92376, +0.92376);
        set(2, "Pmax", -0.92);
        //f->SetParLimits(2,-0.92376, +0.92376);
        set(3, "tau", 4000);
        f->SetParLimits(3,100, 100000);
        set(4, "delta", 0.6432);
       // f->FixParameter(3,1);
        //set(5, "T", 1);
        //set(5,"C", 0);
    };

    static TF1 * GetFunction(std::string name, double T) {
        TF1 * f = new TF1(name.c_str(), ExpJump::root, 0,100000,6);
        InitPars(f,4000);
        f->SetParName(5,"T");
        f->FixParameter(5, T); 
        f->SetNpx(1000);
        return f;
    };

};

class ExpJumpTied {
    double Td;
    double P0;
    double tau0;
    double Pmax;
    double tau;
    double delta;
    public:
    ExpJumpTied(double depol_time, double P0_, double Pmax_, double DELTA_) {
        Td=depol_time;
        P0 = P0_;
        Pmax = Pmax_;
        delta = DELTA_;
        tau0 = 1540./pow(4.1,5.0)*3600.;
        std::cout << tau0 << std::endl;
        double G = Pmax/0.92376;
        tau = fabs(G)*tau0;  
        std::cout <<"G = " << G <<  " tau0 " << tau0  << " tau = " << tau << std::endl;
    }

    double operator() (double t) {
        if (t < Td) return polarization(P0, Pmax, tau, t);
        double P1 = delta + polarization(P0, Pmax, tau, Td); //polarization just after the depolarization
        return polarization(P1, Pmax, tau, t-Td);
    };

    static double root(double *x, double *p) {
        ExpJumpTied fun(p[0], p[1], p[2], p[3]);
        double T= p[4];
        double result =  dgaus(fun, x[0]-T/2, x[0]+T/2, 1e-8)/T;
        //std::cout << p[0] << " " << p[1] << "  " << p[2] << "  " << p[3] <<  "  " << p[4] << "   result = " << result << std::endl;
        return result;
    };

    static void InitPars(TF1 * f, double td) {
        auto set = [&](int i, const char * name, double value) {
            f->SetParName(i,name);
            f->SetParameter(i,value);
        };
        set(0,"Td", td);
        f->SetParLimits(0,0, 100000);
        //f->FixParameter(0, 14000);
        set(1, "P0", -0.923);
        f->SetParLimits(1,-0.92376, +0.92376);
        //set(2, "tau0", 1540*3600/pow(4.1,5.0));
        set(2, "Pmax", -0.92375);
        f->SetParLimits(2,-0.92376, +0.92376);
        //f->FixParameter(2, 1540*3600/pow(4.1,5.0));
        //set(3, "tau", 4000);
        set(3, "delta", 0.6432);
       // f->FixParameter(3,1);
        //set(4, "T", 1);
    };

    static TF1 * GetFunction(std::string name, double T) {
        TF1 * f = new TF1(name.c_str(), ExpJumpTied::root, 0,10000,5);
        InitPars(f,3000);
        f->SetParName(4,"T");
        f->FixParameter(4, T); 
        f->SetNpx(1000);
        return f;
    };


};

std::map<std::string, std::shared_ptr<TGraphErrors>  > GM;
double GLOBAL_TIME_OFFSET;

std::tuple<double, double> Fit3(TGraphErrors * g, double T) {
    TF1 * f = ExpJump::GetFunction("expjump",T);
    //f->SetRange(0,4500);

    double chi2=1e10;
    int best_point=0;
    double best_time = 14000;

    for(int i=2;i<g->GetN();++i) {
        double t =  g->GetX()[i]+T/2;
        //std::cout << i << std::endl;
        ExpJump::InitPars(f, t);
        auto fr = g->Fit("expjump","EX0SQ");
        if(fr->IsValid() && f->GetChisquare() < chi2) {
            best_point = i;
            best_time = t;
            chi2 = f->GetChisquare() ;
        }
    }

    //TRandom R;
    //for (int i=0;i<100;++i) {
    //    double t = R.Uniform(3600,5400);
    //    ExpJump::InitPars(f, t);
    //    auto fr = g->Fit("expjump","EX0S");
    //    double lchi2 = f->GetChisquare();
    //    if(fr->IsValid() && lchi2 < chi2) {
    //        best_time = t;
    //        chi2 = lchi2 ;
    //    }
    //}

    ExpJump::InitPars(f, best_time);
    g->Fit(f->GetName(),"EX0");
    //g->GetXaxis()->SetTitle("time, s");
    //g->GetYaxis()->SetTitle("P");

    double Td = f->GetParameter(0);
    double dTd = f->GetParError(0);
    auto graphE = GM["E"];
    double E = graphE->Eval(Td);
    int idx=0;
    double distance_to_close_point=1e10;
    for(int i=0;i<graphE->GetN();i++) {
      if(fabs(graphE->GetX()[i] - Td) < distance_to_close_point) {
        idx=i;
        distance_to_close_point = fabs(graphE->GetX()[i] - Td);
      }
    }
    int idx_min = idx-3 < 0 ? 0 : idx-3;
    int idx_max = idx+3 >= graphE->GetN() ? graphE->GetN()-1 : idx+3;
    TF1 pol1("mypol1", "[0]+[1]*x",graphE->GetX()[idx_min], graphE->GetX()[idx_max]);
    graphE->Fit("mypol1", "QR");
    double speed = pol1.GetParameter(1);
    if(fabs(speed) > 0.1) std::cout << "Wrong scan speed: ";
    else std::cout << " Calculate scan speed: ";
    std::cout <<  speed*1e3 << " keV/s" <<  std::endl;
    std::cout << "Energy: " << E << " +- " << dTd*0.002 << std::endl;
    return {E, dTd*0.002};
};



void Fit(TF1 * f,  TGraphErrors * g, double T) {
    auto tune_f = [&] (double t) {
        f->SetParameter(0, t); //depol time
        f->SetParName(0,"Td");
        f->SetParameter(1, 0); //delta
        f->SetParName(1,"DELTA");
        f->SetParameter(2, 0); //const
        //f->FixParameter(2,0);
        f->SetParName(2,"CONST");
        f->FixParameter(3, 0); //slope1
        f->SetParName(3,"SLOPE1");
        //f->FixParameter(4, 0); //slope2
        f->SetParName(4,"SLOPE2");
        f->FixParameter(5, 1); //tau
        f->SetParName(5,"TAU");
        f->FixParameter(6, 2); //power
        f->SetParName(6,"POWER");
        f->FixParameter(7, T); //point measurement time
        f->SetParName(7,"T");
        f->SetNpx(1000);
    };

    double chi2=1e100;
    int best_point=0;
    double best_time = 100;

    //for(int i=0;i<g->GetN();++i) {
    //    double t = g->GetX()[i];
    //    tune_f(t);
    //    g->Fit("fun","EX0");
    //    double lchi2 = f->GetChisquare();
    //    if( lchi2 < chi2) {
    //        best_point = i;
    //        best_time = t;
    //        std::cout << "lchi2 = " << lchi2 << "  chi2 = " << chi2 << std::endl;
    //        chi2 = lchi2;
    //    }
    //}

    tune_f(4000);
    //tune_f(1800);
    g->Fit(f->GetName(),"EX0");
    g->GetXaxis()->SetTitle("time, s");
    g->GetYaxis()->SetTitle("P");

    double Td = f->GetParameter(0);
    double dTd = f->GetParError(0);
    double E = 4101.00 + (Td-4*60-2042)*0.01;
    double dE = 0.01*dTd;
    std::cout << "Energy: " << E << " +- " << dE << std::endl;
    char buf[1024];
    sprintf(buf," E = %.2f #pm %.2f MeV", E,dE);
    TLatex * lE = new TLatex(500, 0, buf);
    lE->Draw();
    sprintf(buf," H = %.0f Gs", 3954.);
    TLatex * lH = new TLatex(500,-0.2, buf);
    lH->Draw();
    sprintf(buf," #nu_{x}/#nu_{y} = %.3f/%.3f", 0.534, 0.581);
    TLatex * lnu = new TLatex(500,-0.4, buf);
    lnu->Draw();

    double ymin = g->GetMinimum();
    for(int i=0;i<g->GetN();++i) {
        auto text = new TText();
        //text->SetNDF();
        char buf[1024];
        double t = g->GetX()[i];
        if(t<2042) sprintf(buf,"%s","");
        else sprintf(buf,"%.2f", 4101 + (t-4*60 -2042)*0.01);
        text->SetText(t, -1.4, buf);
        text->SetTextSize(0.02);
        text->SetTextAngle(90);
        if((t+T/2)/60 >4 ) {
        text->Draw();
        }
    }
};

//read and updating existing TGraphErrors in  map container  GM
void read_graph(std::string name, time_t start_view_time=0, time_t end_view_time=std::numeric_limits<time_t>::max()) {
    std::ifstream ifs(name);
    int n;
    std::string n_str;
    double unixtime, E,F, P,dP,Q, dQ, V,dV, beta, dbeta, chi2;
    double dip_amp, dip_ang, quad_amp, quad_ang;
    double fft1_amp, fft1_ang, fft2_amp, fft2_ang;
    double gross_moments, d_gross_moments;
    double n_evt_l, n_evt_r, mx_l, mx_r,  my_l,  my_r, sx_l, sx_r, sy_l, sy_r;
    double asym_x, asym_y;
    double dasym_x, dasym_y;
    auto & g = GM;
    std::string line;
    std::optional<double> global_time_offset; //the time of the first point in the file
    std::optional<double> local_time_offset; //the time of the first point in the file
    int i{0};
    while( std::getline(ifs, line)) {
        if ( line[0]=='#') {
          continue;
        } else {
        }
        std::istringstream iss(line);
        iss >> n >> unixtime >> F >> E>> P >> dP >>Q >>dQ >> V>> dV;
        iss >> beta >> dbeta >> chi2;
        iss >> dip_amp >> dip_ang >> quad_amp >>quad_ang;
        iss >> fft1_amp >> fft1_ang >> fft2_amp >> fft2_ang;
        iss >> gross_moments >> d_gross_moments;
        iss >> n_evt_l >> n_evt_r >> mx_l >> mx_r >> my_l >> my_r;
        iss >> sx_l >> sx_r >> sy_l >> sy_r;
        asym_x = mx_l - mx_r;
        if(n_evt_l==0) n_evt_l=1;
        if(n_evt_r==0) n_evt_r=1;
        dasym_x = std::hypot(sx_l/sqrt(n_evt_l), sx_r/sqrt(n_evt_r));
        asym_y = my_l - my_r;
        dasym_y = std::hypot(sy_l/sqrt(n_evt_l), sy_r/sqrt(n_evt_r));

        if ( ! global_time_offset ) global_time_offset = unixtime;

        if( n>= 0) {
          double t = unixtime - global_time_offset.value();
          if(t  >= start_view_time && t<= end_view_time) {
            if ( ! local_time_offset ) local_time_offset = unixtime;
            double t  = unixtime - local_time_offset.value();
            //function to set point for graph graph_name
            auto set_point = [&](std::string graph_name, double Y, double dY=0) 
            {
              auto graph = [&]() { //this graph always exists
                auto it = g.find(graph_name);
                if(it==g.end()) {
                  auto graph = new TGraphErrors;
                  g[graph_name].reset(graph);
                  return graph;
                } else {
                  return (it->second).get();
                }
              }();
              graph->SetPoint(i, t, Y);
              graph->SetPointError(i, 0, dY);
              graph->SetLineWidth(2);
            };

            set_point("unixtime" , unixtime             , 0);
            set_point("P"        , P             , dP);
            set_point("Q"        , fabs(Q)       , dQ);
            set_point("V"        , V             , dV);
            set_point("beta"     , beta          , dbeta);
            set_point("chi2"     , chi2          , 0);
            set_point("dip_amp"  , dip_amp       , 0);
            set_point("dip_ang"  , dip_ang       , 0);
            set_point("quad_amp" , quad_amp      , 0);
            set_point("quad_ang" , quad_ang      , 0);
            set_point("fft1_amp" , fft1_amp      , 0);
            set_point("fft1_ang" , fft1_ang      , 0);
            set_point("fft2_amp" , fft2_amp      , 0);
            set_point("fft2_ang" , fft2_ang      , 0);
            set_point("GM"       , gross_moments , d_gross_moments);
            set_point("n_evt_l"  , n_evt_l       , sqrt(n_evt_l));
            set_point("n_evt_r"  , n_evt_r       , sqrt(n_evt_r));
            set_point("mx_l"     , mx_l          , sx_l/sqrt(n_evt_l));
            set_point("mx_r"     , mx_r          , sx_r/sqrt(n_evt_r));
            set_point("my_l"     , my_l          , sy_l/sqrt(n_evt_l));
            set_point("my_r"     , my_r          , sy_r/sqrt(n_evt_r));
            set_point("sx_l"     , sx_l          , 0);
            set_point("sx_r"     , sx_r          , 0);
            set_point("sy_l"     , sy_l          , 0);
            set_point("sy_r"     , sy_r          , 0);
            set_point("asym_x"   , asym_x        , dasym_x);
            set_point("asym_y"   , asym_y        , dasym_y);
            set_point("E"        , E             , 0);
            set_point("F"        , F             , 0);
            ++i;
          }
        }
    }
    std::cout << "Read " << i << " points\n";
    std::cout << flush;
    //GLOBAL_TIME_OFFSET = global_time_offset.value();
    GLOBAL_TIME_OFFSET = local_time_offset.value();
    return g;
}

std::map<std::string, std::unique_ptr<TCanvas>> CanvasMap;

static int CANVAS_IDX=0;

struct FitConfig_t {
  int run{0};
  std::chrono::seconds update_time=60s;
  double count_time=300;
  time_t start_view_time = 0;
  time_t end_view_time = std::numeric_limits<time_t>::max();
  std::vector<std::string> draw_list={"P","Q"};
  std::string title;
};

TLatex * ENERGY_LATEX{nullptr};

std::list<std::unique_ptr<TLatex>> EnergyNoteList;

void fit_single(std::string file_name, const FitConfig_t & cfg){
    auto f  = new TF1("fun", &experf,  0, 10000, 8);
    auto fun_pol  = new TF1("fun_pol", "[0]*(1-exp(-(x-[1])/[2]))",  0, 10000);
    fun_pol->SetParameter(0,1);
    fun_pol->SetParameter(1,0);
    fun_pol->SetParameter(2,1000);

    fun_pol->FixParameter(1,0);

    read_graph(file_name, cfg.start_view_time, cfg.end_view_time);
    if(GM.empty()) return;

    double t0 = GM["unixtime"]->GetY()[0];

    auto draw_label = [](double x, double y, const char * format, auto ... args) -> TLatex * {
      char buf[1024];
      std::cout << "Before sprintf" << std::endl;
      snprintf(buf,sizeof(buf),format, args...);
      std::cout << "After sprintf" << std::endl;
      TLatex *  l = new TLatex(x, y,buf); 
      std::cout << "After new TLatex" << std::endl;
      l->SetNDC();
      std::cout << "After SetNDC" << std::endl;
      std::cout << "Before draw" << std::endl;
      l->Draw();
      std::cout << "After draw" << std::endl;
      //std::cout << "Before settextsize" << std::endl;
      //l->SetTextSize(0.04);
      //std::cout << "After settextsize" << std::endl;
      return l;
    };

    auto draw = [&](std::string graph_name) {
      try { 
        auto g = GM.at(graph_name).get();
        TCanvas * c = [&]() { //This canvas always exists
          auto it = CanvasMap.find(graph_name);
          if( it  == CanvasMap.end() ) {
            c = new TCanvas(graph_name.c_str(),graph_name.c_str(), 327,714, 615,365);
            c->SetGridx();
            c->SetGridy();
            CanvasMap[graph_name].reset(c); 
            CANVAS_IDX++;
            std::cout << "Drawing graph " <<  graph_name << std::endl;
            g->Draw("ap");
            g->SetMarkerStyle(20);

            time_t global_time_offset = time_t(GLOBAL_TIME_OFFSET);
            time_t time_end = time_t(GM["unixtime"]->GetY()[GM["unixtime"]->GetN()-1]);
            auto timeinfo_begin = *localtime(&global_time_offset);
            auto timeinfo_end  =  *localtime(&time_end);

            auto xaxis = g->GetXaxis();
            xaxis->SetTitle("time, s");
            xaxis->SetTitleOffset(1.2);
            xaxis->SetTimeOffset(GLOBAL_TIME_OFFSET);
            xaxis->SetTimeDisplay(kTRUE);
            xaxis->SetTimeFormat("%H:%M");

            g->GetYaxis()->SetTitle(graph_name.c_str());
            g->GetYaxis()->CenterTitle();
            g->SetTitle(cfg.title.c_str());

            char datebuf[1024];
            strftime(datebuf,1024, "%Y-%m-%d %H:%M:%S", &timeinfo_begin); 
            auto lDate = new TLatex(0.01, 0.96,datebuf); 
            lDate->SetNDC();
            lDate->SetTextSize(0.04);
            lDate->Draw();

            std::cout << "BEfore draw run " << std::endl;
            if( cfg.run>0) {
              draw_label(0.01,0.91,"Run %d", cfg.run);
            } 
            std::cout << "After draw run " << std::endl;

            std::ifstream ifs("/mnt/vepp4/kadrs/nmr.dat");
            ifs.ignore(65535,'\n');
            double H;
            ifs >> H;
            ifs.close();
            draw_label(0.564, 0.0189, "H = %8.3f Gs", H);

            return c;
          } else {
            auto c = it->second.get();
            c->cd();
            std::cout << "Updating graph " <<  graph_name << std::endl;
            return c;
          }
        }();
        
        if(graph_name=="P") {
          gStyle->SetOptFit();
          auto [E,dE] = Fit3(g,cfg.count_time);
          if ( E > 1 ) {
            if ( ENERGY_LATEX ) delete ENERGY_LATEX;
            ENERGY_LATEX = draw_label(0.01, 0.0189, "E = %8.3f #pm %4.3f MeV", E, dE);
          } 
        }
        c->Modified();
        c->Update();
      }
      catch(...) {
        std::cerr << "fit: fitsingle: Something wrong in draw lambda function\n";
      }
    };

    for(auto it=cfg.draw_list.rbegin(); it!=cfg.draw_list.rend(); ++it){
      draw(*it);
    }

    EnergyNoteList.clear();
    if(auto itP = GM.find("P"); itP!=GM.end()) {
      auto & Pg = itP->second;
      if(auto itE = GM.find("E"); itE!=GM.end()) {
        auto & Eg = itE->second;
        auto itC = CanvasMap.find("P");
        if( itC  != CanvasMap.end() ) {
          for( int i=0; i<Pg->GetN(); ++i) {
            char buf[1024];
            if(Eg->GetY()[i]>100) {
              snprintf(buf,1024,"%-7.2f", Eg->GetY()[i]);
            } else {
              snprintf(buf,1024,"%s", ".");
            }
            auto l = new TLatex(Eg->GetX()[i],Pg->GetHistogram()->GetMinimum(),buf);
            l->SetTextAngle(90);
            l->SetTextSize(0.04);
            EnergyNoteList.push_back( std::unique_ptr<TLatex>(l));
            itC->second->cd();
            l->Draw();
          }
        }
      }
    }
}

void graph_list(void) {
  for(const auto & [name, g] : GM) {
    std::cout << name << "\n";
  }
  std::cout << std::flush;
}

void fitloop(std::string name, FitConfig_t cfg=FitConfig_t()) {
  std::cout << "Reading file " << name << std::endl;
  GM.clear();
  CanvasMap.clear();
  using namespace std::chrono;
  if ( cfg.end_view_time <= cfg.start_view_time ) {
    std::cerr << "ERROR: wrong time range: end time less or equal start time\n";
  }
  struct stat statbuf;
  time_t last_update=0;
  while(true) {
    fit_single(name, cfg);
    auto tb = std::chrono::system_clock::now();
    do {
      int rc = stat(name.c_str(), &statbuf);
      gSystem->ProcessEvents();
      this_thread::sleep_for(100ms);
    } while (last_update == statbuf.st_mtim.tv_sec);
    last_update = statbuf.st_mtim.tv_sec;
    //while( std::chrono::system_clock::now() < tb + cfg.update_time) {
    //while( last_update == ) {
    //  gSystem->ProcessEvents();
    //  this_thread::sleep_for(100ms);
    //  std::cout << statbuf.st_mtim << std::endl;
    //}
  }
};

void fitloop(time_t start_view_time=0, time_t end_view_time=std::numeric_limits<time_t>::max()) {
  FitConfig_t cfg;
  cfg.update_time = 60s;
  cfg.count_time = 300;
  cfg.start_view_time = start_view_time;
  cfg.end_view_time = end_view_time;
  fitloop("/home/lsrp/polarimeter_dev/tmp/pol_fitres.txt", cfg);
}

