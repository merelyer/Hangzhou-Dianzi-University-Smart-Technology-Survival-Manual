a=0.8;
difference()
{
    translate([0,0,50/2]) cube([50,50,50],center=true);
    translate([0,0,0]) cube([50-2*a,50-2*a,120],center=true);
}
