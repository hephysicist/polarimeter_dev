
#include <chrono>
#include <time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <math.h>

R__LOAD_LIBRARY(libfmt.so)
#include <fmt/format.h>



//algorithm for integrating function func in the range (a,b) with accuracy epsilon
template <class F> double dgaus(F & func, double a, double b, double epsilon)
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


double polarization(double P, double Pmax, double tau, double t) {
    return P + (Pmax-P) * (1.0 - exp(- t/tau));
}

class MultiExpJump {

    const int Njumps; //number of depolarization jumps
    std::vector<double> Tds;    //polarization times in seconds  Tds[0]=0 - fixed, another  Tds[1,2,Njumps] are free.
    std::vector<double> Deltas; //depolarization jump values (free) 0,1,...Njump-1
    std::vector<double> Fs; //buffer
    std::vector<double> Ps; //Ps[0] - minimization parameter, other is the buffer
    double Pmax; //Maximum polarization (free)
    double tau; //polarization time (Sokolov Ternov) (free)
    int Npars; //total number of fit parameters
    double Taud; //depolarization time depends on depolarizer power (fixed)
    double T;// count time (usualy 300 sec) (fixed)

    public:

    MultiExpJump(int n, double Tc, double taud) : Njumps(n), T(Tc), Taud(taud) {
      n = Njumps+1;
      //create buffers
      Tds.resize(n);
      Deltas.resize(n);
      Ps.resize(n);
      Fs.resize(n);
      Npars = 3 + 2*Njumps;
    };

    int GetNpars() const {return Npars; } 

    int GetNjumps() const {return Njumps; } 

    double operator() (double t) {
        //Initial value must be set. This is not minimization parameter.
        Tds[0]    = 0;

        //jump times must be sorted for correct work of the algorithm
        std::sort(std::begin(Tds), std::end(Tds));

        //just alias to shorten polarization call
        auto pol = [this] (double P0, double tt) -> double {
          return polarization(P0, Pmax, tau, tt);
        };

        //Heaviside like function
        auto H = [this](double tt, double tau_d) {
          return 0.5*(1. + erf(tt/tau_d));
        };

        //calculate part of polarization curves between jumps
        for(int i=0;i<=Njumps; ++i) {
          Fs[i] = pol(Ps[i], t-Tds[i]);
          //calculate initial polarization value for next curve
          if(i<Njumps) Ps[i+1] = Deltas[i] + pol(Ps[i], Tds[i+1]-Tds[i]);
        }

        //accumulate the polarization
        double P{Fs[0]};
        for(int i=0;i<Njumps; ++i) {
          auto h = H(t - Tds[i+1],Taud); //jumps are smooth
          P += (Fs[i+1] - Fs[i]) * h; //describes smooth transition between behavior before and after depolarization jump Tds[i+1]
        }
        return P;
    };

    //this function is used by TF1
    double operator() ( double *x, double *p) {
      double t = x[0];
      tau = p[0];
      Pmax = p[1];
      Ps[0] = p[2];
      int idx=3;
      for(int i = 0; i<Njumps; ++i, ++idx) {
        Tds[i+1] = p[idx]; //jump position 
      } 
      for(int i = 0; i<Njumps; ++i, ++idx) {
        Deltas[i] = p[idx]; //jump amplitude
      }
      return dgaus(*this,t, t+T, 1e-8)/T; //calculate average 
    }
};


std::map<std::string, std::shared_ptr<TGraphErrors>  > GM;
double GLOBAL_TIME_OFFSET;

struct FitConfig_t {
  int run  = 0;
  //std::chrono::seconds update_time=60s;
  double count_time=300;
  double speed = 0.01; //scan speed MeV/sec. The sign shows scan direction
  time_t start_view_time = 0;
  time_t end_view_time = std::numeric_limits<time_t>::max();
  std::vector<std::string> draw_list={"P","Q"};
  std::string title; //graph title
  std::string save_dir="/home/lsrp/Measurements";
  bool save = false;
  std::vector<double> Tds; //initial values for jumps to help minimization function find good minimum
  int Njumps = 1; //numer of jumps in the fit
  double taud = 5; //The depolarization characteristic time (smooth jump by erf function)  in seconds
};

std::vector<double> NMRs;

//calculte the energy from fit result for parameter n
auto get_energy( TF1 * f, const FitConfig_t & cfg, int n) -> std::tuple<double, double> {
  double Td = f->GetParameter(n);
  double dTd = f->GetParError(n);
  NMRs.push_back(GM["H"]->Eval(Td));
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
  int idx_min = idx-1 < 0 ? 0 : idx-1;
  int idx_max = idx+1 >= graphE->GetN() ? graphE->GetN()-1 : idx+1;
  TF1 pol1("mypol1", "[0]+[1]*x",graphE->GetX()[idx_min], graphE->GetX()[idx_max]);
  graphE->Fit("mypol1", "QR");
  double speed = pol1.GetParameter(1);
  if(fabs(speed) > 0.1) {
    std::cout << "Wrong scan speed: " << speed*1e3 << " keV/s."  << " Use default scan speed: ";
    speed = cfg.speed;
  }
  else std::cout << "Calculated scan speed: ";
  std::cout <<  fabs(speed*1e3) << " keV/s" <<  std::endl;
  std::cout << fmt::format("Energy: {:8.3f} +- {:4.3f} MeV", E, dTd*speed) << std::endl;
  if(idx>0) {
    if(graphE->GetY()[idx-1]==0) {
      E=graphE->GetY()[idx];
    }
  }
  return {E, fabs(dTd*speed)};
};



struct FitResult {
  double t, dt;
  double E, dE;
  double H;
};
//Main fit function
std::vector<FitResult> FitGraph(TGraphErrors * g, const FitConfig_t & cfg) {

  auto fun = new MultiExpJump(cfg.Njumps, cfg.count_time, cfg.taud);

  TF1 * f = new TF1("multi_exp_jump", *fun, 0, 1, fun->GetNpars());
  f->SetNpx(500);

  auto initpar = [&](int i, std::string name, double value, bool fix=false) {
    f->SetParName(i,name.c_str());
    f->SetParameter(i,value);
    if(fix) { f->FixParameter(i,value); }
  };

  initpar(0, "tau", 1500);
  initpar(1, "Pmax", -1);
  initpar(2, "P0", 0);

  const int TdJumpIdx=3;
  int idx=TdJumpIdx;
  for(int i=0;i<fun->GetNjumps();++i,++idx) {
    initpar(idx, "Td"+std::to_string(i+1), 0);
  }
  for(int i=0;i<fun->GetNjumps();++i, ++idx) {
    initpar(idx, "Delta"+std::to_string(i+1), 0);
  }


  //Looking for best chi2
  if( cfg.Tds.empty() ) {
    double chi2=1e100;
    for(int njump=0;njump<fun->GetNjumps();++njump) {
      int best_point=0;
      double best_time = 0;
      for(int i=3;i<g->GetN()-1;++i) {
        double t =  g->GetX()[i]+cfg.count_time/2;
        std::cout << i << " " << t <<  std::endl;
        f->SetParameter(TdJumpIdx+njump,t);
        auto fr = g->Fit("multi_exp_jump","0QEX0S");
        // options
        // 0 - do not draw result
        // Q - minimum printing
        // EX0 - do not take into account errors on axis X (if exist)
        // S  - return FitResult
        if(fr->IsValid() && f->GetChisquare() < chi2) {
          best_point = i;
          best_time = t;
          chi2 = f->GetChisquare();
          //std::cout << "Find better chi2: "<< chi2 << " for point "  << best_point << " and time " << best_time << std::endl;
        }
      }
      std::cout << "Found best time for jump << " << njump << ": " << best_time << std::endl;
      f->FixParameter(TdJumpIdx+njump, best_time);
    }
    //Release fixed parameters
    for(int njump=0;njump<fun->GetNjumps();++njump) {
      f->ReleaseParameter(njump+TdJumpIdx);
    }
  } 
  else if(cfg.Tds.size() <= cfg.Njumps) { //Use user preset values instead
    for(int njump=0;njump<cfg.Tds.size(); ++njump) {
        f->SetParameter(TdJumpIdx+njump,cfg.Tds[njump]);
    }
  }

  //main fit
  g->Fit("multi_exp_jump","EX0");

  //prepare multiple jumps for future display on canvas. Doesnt work yet
  std::vector<FitResult> R;
  for(int i=0;i<fun->GetNjumps(); ++i) {
    FitResult fr;
    fr.t = f->GetParameter(i+TdJumpIdx);
    fr.dt = f->GetParError(i+TdJumpIdx);
    std::tie(fr.E, fr.dE) = get_energy(f,cfg, i+TdJumpIdx);
    fr.H = GM["H"]->Eval(fr.t);
    R.push_back(fr);
  }
  return R;
};

//read and updating existing TGraphErrors in  map container  GM
void read_graph(std::string name, time_t start_view_time=0, time_t end_view_time=std::numeric_limits<time_t>::max()) {
    std::cout << "Reading file " << name << std::endl;
    std::ifstream ifs(name);
    if(!ifs) {
      std::cerr << "Unalbe to read file " << name << std::endl;
      return;
    }
    int n;
    std::string n_str;
    double unixtime, H, E,F, P,dP,Q, dQ, V,dV, beta, dbeta, chi2;
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
        iss >> n >> unixtime >> F >> E >> H >> P >> dP >>Q >>dQ >> V>> dV;
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
            set_point("H"        , H             , 0);
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

struct Canvas {
  std::unique_ptr<TCanvas> canvas; //The TCanvas
  std::vector<std::unique_ptr<TLatex>> energy_text; //Energy information
  std::vector<std::unique_ptr<TLatex>> nmr_text; //nmr text
  std::list<std::unique_ptr<TLatex>> EnergyNoteList;
};

//std::map<std::string, std::unique_ptr<TCanvas>> CanvasMap;
std::map<std::string, Canvas> CanvasMap;

static int CANVAS_IDX=0;


//TLatex * ENERGY_LATEX{nullptr};
std::vector<std::unique_ptr<TLatex>> ENERGY_LATEX;
std::vector<std::unique_ptr<TLatex>> NMR_LATEX;

std::list<std::unique_ptr<TLatex>> EnergyNoteList;

void fit_single(std::string file_name, const FitConfig_t & cfg){
    read_graph(file_name, cfg.start_view_time, cfg.end_view_time);
    if(GM.empty()) return;

    double t0 = GM["unixtime"]->GetY()[0];

    auto draw_label = [](double x, double y, const char * format, auto ... args) -> TLatex * {
      char buf[1024];
      snprintf(buf,sizeof(buf),format, args...);
      TLatex *  l = new TLatex(x, y,buf); 
      l->SetNDC();
      l->Draw();
      //l->SetTextSize(0.04); //This line result in program crash (segmentation violation)
      return l;
    };

    auto draw = [&](std::string graph_name) {
      try { 
        auto g = GM.at(graph_name).get();
        Canvas * cnvs = [&]() { //This canvas always exists
          auto it = CanvasMap.find(graph_name);
          if( it  == CanvasMap.end() ) {
            //c = new TCanvas(graph_name.c_str(),graph_name.c_str(), 327,714, 615,365);
            TCanvas * c = new TCanvas(graph_name.c_str(),graph_name.c_str(), 838,570,1022,459);
            c->SetGridx();
            c->SetGridy();
//            c->SetPad(0,0,0.8,0.8);
            CanvasMap[graph_name].canvas.reset(c); 
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
            //g->SetTitle(cfg.title);

            char datebuf[1024];
            strftime(datebuf,1024, "%Y-%m-%d %H:%M:%S", &timeinfo_begin); 
            auto lDate = new TLatex(0.01, 0.96,datebuf); 
            lDate->SetNDC();
            lDate->SetTextSize(0.04);
            lDate->Draw();

            //auto Hlatex = draw_label(0.564, 0.0189, "H = %8.3f Gs", GM["H"]->GetY()[0]);
            //Hlatex->SetTextSize(0.04);
            ////it->second.nmr_text.push_back(std::unique_ptr<TLatex>(Hlatex));
            //NMR_LATEX.reset(Hlatex);

            if( cfg.run>0) {
              auto l = draw_label(0.01,0.91,"Run %d", cfg.run);
              l->SetTextSize(0.04);
            } 
            return &CanvasMap[graph_name];
          } else {
            auto c = it->second.canvas.get();
            c->cd();
            std::cout << "Updating graph " <<  graph_name << std::endl;
            return &(it->second);
          }
        }();
        
        if(graph_name=="P") {
          std::cout << "Fitting graph P" << std::endl;
          gStyle->SetOptFit();
          NMRs.clear();
          std::vector<FitResult> FR = FitGraph(g,cfg);
          ENERGY_LATEX.clear();
          NMR_LATEX.clear();
          double x = 0.01;
          double y_NMR=0.35;
          NMR_LATEX.push_back(std::unique_ptr<TLatex>(new TLatex)); 
          NMR_LATEX.back()->SetText(0.9,y_NMR, fmt::format("H_{{0}} = {:8.3f} Gs", GM["H"]->GetY()[0]).c_str());
          NMR_LATEX.back()->SetNDC();
          NMR_LATEX.back()->Draw();
          NMR_LATEX.back()->SetTextSize(0.03);
          y_NMR-=0.07;

          int idx=1;
          for(auto & fr : FR) {
            if ( fr.E > 1 ) {
              ENERGY_LATEX.push_back(std::unique_ptr<TLatex>(draw_label(x, 0.0189, fmt::format("E_{{{}}} = %8.3f #pm %4.3f MeV", idx).c_str(), fr.E, fr.dE)));
              ENERGY_LATEX.back()->SetTextSize(0.04);
              x+=0.24;
            }
            NMR_LATEX.push_back(std::unique_ptr<TLatex>(new TLatex)); 
            NMR_LATEX.back()->SetText(0.9,y_NMR, fmt::format("H_{{{}}} = {:8.3f} Gs",idx, fr.H).c_str());
            NMR_LATEX.back()->SetNDC();
            NMR_LATEX.back()->Draw();
            NMR_LATEX.back()->SetTextSize(0.03);
            y_NMR-=0.07;
            idx++;
          }
          if(cfg.save) {
            char date[1024];
            time_t global_time_offset = time_t(GLOBAL_TIME_OFFSET);
            auto timeinfo_begin = *localtime(&global_time_offset);
            strftime(date, sizeof(date), "%Y-%m-%dT%H-%M-%S", &timeinfo_begin);
            char filename[65535];
            auto save = [&](const char * type) {
              snprintf(filename, 65535, "%s/R%04d-%s.%s", cfg.save_dir.c_str(), cfg.run, date, type);
              cnvs->canvas->SaveAs(filename);
            };
            save("pdf");
            save("root");
            save("png");
          }
        }
        cnvs->canvas->Modified();
        cnvs->canvas->Update();
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
            auto l = new TLatex(Eg->GetX()[i], Pg->GetHistogram()->GetMinimum(),
                Eg->GetY()[i]>100 ? 
                fmt::format("{:<7.2f}", Eg->GetY()[i]).c_str() :
                ".");
            l->SetTextAngle(90);
            l->SetTextSize(0.04);
            EnergyNoteList.push_back( std::unique_ptr<TLatex>(l));
            itC->second.canvas->cd();
            l->Draw();
          }
          itC->second.canvas->Modified();
          itC->second.canvas->Update();
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

void fitloop(std::string filename, FitConfig_t cfg=FitConfig_t()) {
  std::clog << "Clear old data" << std::endl;
  GM.clear();
  CanvasMap.clear();
  if ( cfg.end_view_time <= cfg.start_view_time ) {
    std::cerr << "ERROR: wrong time range: end time less or equal start time\n";
  }
  struct stat statbuf;
  time_t last_update=0;
  while(true) {
    fit_single(filename, cfg);
    auto tb = std::chrono::system_clock::now();
    do {
      int rc = stat(filename.c_str(), &statbuf);
      gSystem->ProcessEvents();
      this_thread::sleep_for(100ms);
    } while (last_update == statbuf.st_mtim.tv_sec);
    last_update = statbuf.st_mtim.tv_sec;
  }
};

void fitloop(time_t start_view_time=0, time_t end_view_time=std::numeric_limits<time_t>::max()) {
  FitConfig_t cfg;
  cfg.count_time = 300;
  cfg.start_view_time = start_view_time;
  cfg.end_view_time = end_view_time;
  fitloop("/home/lsrp/polarimeter_dev/tmp/pol_fitres.txt", cfg);
}

