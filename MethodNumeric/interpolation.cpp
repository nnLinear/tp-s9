#include <iostream>
#include <map>

using namespace std;


map<double , double> fn = {
    {3 , 5},
    {4 , 7},
    {5 , 10},
    {6 , 6},
    {7 , 13},
    {8 , 20}
};

const double X = 1;

double Li(double , double);
double Pn(double  , const map<double , double> );

int main (){
    cout << X << " : " << Pn(X , fn) << endl;
    
    return 0;
}


double Li(double x , double xi){
    double ans=1;
    for (const auto& point : fn){
        if(xi != point.first){
            ans *= (x-point.first)/(double)(xi-point.first);
        }
    }
    return ans;
}

double Pn(double x ,  const map<double , double> fn){
    double ans = 0;
    for (const auto& podouble : fn){
        ans += podouble.second * Li(x , podouble.first);
    }
    return ans;
}