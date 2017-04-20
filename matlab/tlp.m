function b1 = tlp(Y, X, wt, netwk, b0, delta1, delta2, tau1, tau2)
    [b1, s1] = cvxtlp(Y, X, wt, netwk, b0, delta1, delta2, tau1, tau2);
    [b2, s2] = cvxtlp(Y, X, wt, netwk, b1, delta1, delta2, tau1, tau2);
    while (s1-s2>0.001)
        s1=s2;
        b1=b2;
        [b2, s2] = cvxtlp(Y, X, wt, netwk, b1, delta1, delta2, tau1, tau2);
    end
    b1=b2;
end

