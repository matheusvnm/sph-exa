$benchnum  = '632';
$benchname = 'sph_exa_s';
$exename   = 'sph_exa';
$benchlang = 'CXX';
@base_exe  = ($exename);

$reltol = 0.006;
$abstol = 0.0000004;

@sources = qw( 
sqpatch.cpp
               );

$need_math = 'no';

$bench_flags = '';
$bench_cxxflags = '-DUSE_MPI -DSPEC_USE_LT_IN_KERNELS -I. -Iinclude -Iinclude/tree ';

sub invoke {
    my ($me) = @_;
    my $name;
    my @rc;
    my @temp = main::read_file('control');
    my $exe = $me->exe_file;
    for my $line (@temp) {
    chomp $line;
    next if $line =~ m/^\s*(?:#|$)/;
    push (@rc, { 'command' => $exe, 
                    'args'    => [ $line ], 
                    'output'  => "$exename.out",
                    'error'   => "$exename.err",
                });
    }
    return @rc;
}

%deps = ();


sub pre_build {
        my ($me, $dir, $setup, $target) = @_;
        my $bname = "$benchnum\.$benchname";
        my $pmodel = $me->pmodel;
        $me->pre_build_set_pmodel($dir, $setup, $target, $bname);
        return 0;
}
1;
