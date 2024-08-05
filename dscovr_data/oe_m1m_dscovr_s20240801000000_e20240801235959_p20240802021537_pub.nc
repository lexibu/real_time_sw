CDF  �   
      time          *   Conventions       ACDD-1.3, Spase v2.2.3     title         /DSCOVR Magnetometer Level 2 One Minute Averages    id        Doe_m1m_dscovr_s20240801000000_e20240801235959_p20240802021537_pub.nc   naming_authority      gov.noaa.swpc      program       DSCOVR     summary       }Interplanetary magnetic field observations collected from magnetometer on DSCOVR satellite - 1-minute average of Level 1 data      keywords      _NumericalData.ObservedRegion.Heliosphere.NearEarth, NumericalData.MeasurementType.MagneticField    keywords_vocabulary       Spase v2.2.2   
references        �; DSCOVR TIME SERIES DATA AVERAGES ALGORITHM THEORETICAL BASIS DOCUMENT, v2.4; GSE TO GSM COORDINATE TRANSFORMATION ALGORITHM THEORETICAL BASIS DOCUMENT v2.1      metadata_link         �http://www.ngdc.noaa.gov/docucomp/page?xml=NOAA/NESDIS/NGDC/STP/Space_Weather/iso/xml/satellite-systems_dscovr.xml&view=getDataView&header=none    license       Spase.Access Rights.Open   institution       NOAA   source        DSCOVR Magnetometer Level 1    platform      'Deep Space Climate Observatory (DSCOVR)    
instrument        +boom-mounted triaxial fluxgate magnetometer    history       ,DSCOVR real-time telemetry processing system   	algorithm         FDSCOVR MAGNETOMETER LEVEL 1B DATA ALGORITHM THEORETICAL BASIS DOCUMENT     algorithmVersion      B      algorithmDate         
2015-10-15     processing_level      Level 2    processing_level_description      11-minute average using Hodges-Lehmann M-estimator      date_created      2024-08-02T02:15:37.532Z   date_calibration_data_updated         2024-05-06T00:00:00.000Z   time_coverage_duration        P01D   time_coverage_start       2024-08-01T00:00:00.000Z   time_coverage_end         2024-08-01T23:59:59.000Z   time_coverage_resolution      PT1M   creator_name      Doug Biesecker     creator_type      person     creator_institution       DOC/NOAA/NWS/NCEP/SWPC     creator_email         doug.biesecker@noaa.gov    creator_url       http://www.swpc.noaa.gov/      publisher_name         National Geophysical Data Center   publisher_type        institution    publisher_institution         DOC/NOAA/NESDIS/NGDC   publisher_email       william.rowland@noaa.gov   publisher_url          http://www.ngdc.noaa.gov/dscovr/   records_maximum         �   records_present         �   records_data        �   records_fill             records_missing                    time                description       "date and time for each observation     
short_name        time   C_format      %.13g      units         'milliseconds since 1970-01-01T00:00:00Z    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   standard_name         time   calendar      	gregorian           7   sample_count                description       /number of full resolution measurements averaged    
short_name        sample_count   C_format      %d     units         samples    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max           �        7   measurement_mode                description       7measurement range selection mode (0 = auto, 1 = manual)    
short_name        mode   C_format      %1d    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   measurement_range                   description       5measurement range (~4x sensitivity increase per step)      
short_name        range      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	valid_min                	valid_max                    7   bt               	   description       )Interplanetary Magnetic Field strength Bt      
short_name        bt     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         )bt_interplanetary_magnetic_field_strength      	valid_min                	valid_max                    7    bx_gse               
   description       \Interplanetary Magnetic Field strength Bx component in Geocentric Solar Ecliptic coordinates   
short_name        bx_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7$   by_gse               
   description       \Interplanetary Magnetic Field strength By component in Geocentric Solar Ecliptic coordinates   
short_name        by_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7(   bz_gse               
   description       \Interplanetary Magnetic Field strength Bz component in Geocentric Solar Ecliptic coordinates   
short_name        bz_gse     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gse      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         7,   	theta_gse                	   description       RInterplanetary Magnetic Field clock angle in Geocentric Solar Ecliptic coordinates     
short_name        	theta_gse      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         70   phi_gse              	   description       RInterplanetary Magnetic Field polar angle in Geocentric Solar Ecliptic coordinates     
short_name        phi_gse    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSE         74   bx_gsm               
   description       bInterplanetary Magnetic Field strength Bx component in Geocentric Solar Magnetospheric coordinates     
short_name        bx_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bx_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         78   by_gsm               
   description       bInterplanetary Magnetic Field strength By component in Geocentric Solar Magnetospheric coordinates     
short_name        by_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -by_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7<   bz_gsm               
   description       bInterplanetary Magnetic Field strength Bz component in Geocentric Solar Magnetospheric coordinates     
short_name        bz_gsm     C_format      %.4f   units         nT     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	long_name         -bz_interplanetary_magnetic_field_strength_gsm      	valid_min         ��     	valid_max               _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7@   	theta_gsm                	   description       XInterplanetary Magnetic Field clock angle in Geocentric Solar Magnetospheric coordinates   
short_name        	theta_gsm      C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min         ����   	valid_max            Z   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7D   phi_gsm              	   description       XInterplanetary Magnetic Field polar angle in Geocentric Solar Magnetospheric coordinates   
short_name        phi_gsm    C_format      %.4f   units         degrees    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   missing_value         ��i�       	valid_min                	valid_max           h   _CoordinateSystems        GSpase.NumericalData.Parameter.CoordinateSystem.CoordinateSystemName.GSM         7H   backfill_flag                   description       �One or more measurements were backfilled from the spacecraft recorder and therefore were not available to forecasters in real-time     
short_name        backfill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         backfilled_data_flag   	valid_min                	valid_max                    7L   future_packet_time_flag                 description       rOne or more measurements were extracted from a packet whose timestamp was in the future at the point of processing     
short_name        future_packet_time_flag    C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         packet_time_in_future_flag     	valid_min                	valid_max                    7P   old_packet_time_flag                description       }One or more measurements were extracted from a packet whose timestamp was older than the threshold at the point of processing      
short_name        old_packet_time_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %packet_time_older_than_threshold_flag      	valid_min                	valid_max                    7T   	fill_flag                   description       Fill   
short_name        	fill_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         	fill_flag      	valid_min                	valid_max                    7X   possible_saturation_flag                description       �Possible magnetometer saturation based on a measurement range smaller than the next packet's range or by the mag being in manual range mode.   
short_name        possible_saturation_flag   C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         %possible_magnetometer_saturation_flag      	valid_min                	valid_max                    7\   calibration_mode_flag                   description       Instrument in calibration mode     
short_name        calibration_mode_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         calibration_mode_flag      	valid_min                	valid_max                    7`   maneuver_flag                   description       4AOCS non-science mode (spacecraft maneuver/safehold)   
short_name        maneuver_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         /AOCS_non_science_mode_maneuver_or_safehold_flag    	valid_min                	valid_max                    7d   low_sample_count_flag                   description       $Average sample count below threshold   
short_name        low_sample_count_flag      C_format      %d     lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale   	long_name         )average_sample_count_below_threshold_flag      	valid_min                	valid_max                    7h   overall_quality                 description       ;Overall sample quality (0 = normal, 1 = suspect, 2 = error)    
short_name        overall_quality    C_format      %d     units         n/a    lin_log       TSpase.NumericalData.Parameter.Structure.Element.RenderingHints.ScaleType.LinearScale        7lBy��   "          A1����G��G�������
Cd����G��G�>��H@"�\Cd��                                    By�֦  "          A0Q���(���!G��QG�Cc���(��{>u?��\Cc�                                     By��L  �          A0����G�����{��C_�f��G����
�L����ffC`c�                                    By���  T          A0���33��{�\����{CX@ �33�����1G��h��CY�3                                    By��  �          A1��� ��������z���
=CX�q� �����^{���HCZ��                                    By�>  �          A9��ƸR���5�a�Cmff�ƸR��
>�{?�
=Cms3                                    By��  �          A6�H��p��33��33����Cn�{��p��(����;�Cn�                                     By�.�  "          A:=q�Ǯ����Q��G�CmT{�Ǯ����!G��FffCm��                                    By�=0  �          A:�H��(��{�33�!�Cj��(���Ϳ��\����Ck=q                                    By�K�  �          A7���Q���H�\)�H��Ch�H��Q��ff��G���Ci�=                                    By�Z|  �          A5���=q�=q�?\)�tQ�Clh���=q��\��p�� ��Cm.                                    By�i"  �          A4���Ǯ�{�J=q���
Cj���Ǯ��R�
�H�1G�CkǮ                                    By�w�  "          A4���У��  �z=q��\)Cg�f�У��	��?\)�w33Ch�f                                    By��n  �          A6ff��ff����L������Ci���ff����{�4  Cj��                                    By��  
�          A4Q����
�	��=p��s�Chz����
�=q���R�#
=CiW
                                    By���  T          A-���\)��z�=�\)>\Cf�
��\)���\?p��@�Q�Cfc�                                    By��`  �          A%����z���(�>�=q?�G�Cd�q��z���?�{@�G�Cd�3                                    By��  �          A!�ҏ\��?(��@r�\Cd��ҏ\��?��RA��Cdz�                                    By�Ϭ  �          A  ���R���
?�R@��CdG����R����?��A��CcǮ                                    By��R  
�          A�����R��G�>�@Q�Cb�f���R��ff?�33@��Cb5�                                    By���  �          A���{��
=?5@�33Ca)��{��33?���A�C`�=                                    By���  
�          A����ff��z�@{AqC\����ff��33@3�
A�{C[xR                                    By�
D  T          A ��������@\)A���C]�������@�\)B{CZ�H                                    By��  T          @���������\>\)?�ffC]�������G�?5@�=qC]G�                                    By�'�  �          A((�������녾k����
Cb�������G�?�R@W�Ca�q                                    By�66  �          A!p����H��=q>aG�?��
C\�����H��  ?u@�Q�C\��                                    By�D�  "          A���\)�Å?�A
=CZJ=��\)���@ffAB�HCY^�                                    By�S�  �          A*{�=q���>\)?@  C[��=q��  ?h��@�C[�                                    By�b(  
�          A3�
���z�>�p�?�\)CY������?�33@��
CY�=                                    By�p�  �          A2�R�������R�����G�C`�������H�5�l��Ca@                                     By�t  �          A4����G��   �fff��p�Ce�3��G���*=q�]�Cf�                                    By��  �          A<Q���z���\������\)Cd����z��33��z���
=Cg                                      By���  T          A>�R��33���������
=Cf�\��33�����R��ffCi&f                                    By��f  T          AB{��{��\)��p��(�Ch�q��{�����Q���Ck�                                    By��  "          AB�R������=q�(G�Cp�)���  ����
Cs(�                                    By�Ȳ  �          A@Q���{��33����"
=Ci����{�\)����=qClz�                                    By��X  
�          A<Q���\)�\�33�7��Cc�R��\)�����{�#(�Ch                                    By���  z          A;
=�����G��
�R�?Q�C`�������Q����R�+�Ce��                                    By���  "          A;\)��Q���Q��=q�Ep�C]���Q���Q���
�2�Cb=q                                    By�J  T          A:=q��ff����33�#�RCc8R��ff������G��\)Cf�R                                    By��  "          A9p���33���\����E=qC`�)��33��=q����1\)Ce�q                                    By� �  
�          A6ff��G���  ��R�V�RC`� ��G���G�����B�Cfn                                    By�/<  �          A4z���ff�������L��Ca����ff����{�8Q�Cf                                    By�=�  �          A6{��G����H�p��K�HCaO\��G���33��R�7��CfxR                                    By�L�  �          A7
=��G��z=q�G��m��C^W
��G���������Z{Ce��                                    By�[.  T          A3��~{��
=�=q�l  Cb�=�~{���H���V��Cih�                                    By�i�  
�          A2�\�n{����H�o��CdO\�n{�����Z33Cj�3                                    By�xz  
Z          A0(��y���|���33�n=qCa^��y�������ff�Yp�ChY�                                    By��   �          A$Q��]p���{���CP5��]p��>�R��H�~�C\�                                     By���  �          Aff�n�R�z��޸R�g�CS���n�R�H����(��V��C\!H                                    By��l  �          A
=���H�ᙚ�(��z=qCg�{���H��=q>u?��Cg��                                    By��  �          A���\)������z����Co\�\)����s33����Cp�f                                    By���  �          AG��z=q��(���33�*33Cj  �z=q�\�����{CmE                                    By��^  �          A�
�~{��
=��ff�9��Cg\)�~{��  ��G��#G�Ck\)                                    By��  
�          A�\��G����\�~�R�ҏ\CbxR��G������O\)��
=Cd�\                                    By���  "          A33�����G���{�M�
C`�������z���z��8��CfG�                                    By��P  �          A$(�������=q���[  Cbٚ������������E\)Ch�R                                    By�
�  "          A=q��ff��=q��33��RC^����ff���������ᙚCa�q                                    By��  "          A�\���R��Q�������Cb� ���R���\�����G�Ce�\                                    By�(B  �          @ٙ�������\?0��@��Ca������ff?�G�A3�C`�                                     By�6�  
�          @�p���33��p������RCa� ��33��p���  ��G�Cc.                                    By�E�  "          A�����>�R���Z=qCV^�����x����  �Hz�C]��                                    By�T4  
�          A(������_\)��(��O(�C\\������=q��z��;ffCb0�                                    By�b�  �          @���L(��\)���\�XffCY���L(��J�H��\)�E=qC`�
                                    By�q�  "          @�������5���H�G�CP�������U�l������CT�3                                    By��&  .          @�(��W
=���R��\)���Ci  �W
=��(���ff�%��Cj                                    By���  z          @�(��&ff�`  ������p�Cin�&ff�j=q��  �HQ�Cj��                                    By��r  "          @�z����H�E���G��Q�CU�)���H�fff�w
=�{CZ�                                    By��  
4          @�=q�q��u��,(��ԸRCac��q���z�����{Cc�H                                    By���  z          @Ӆ���
��
=�
�H��33Ca����
���R��=q�\��Cc0�                                    By��d  "          @�=q���������p���z�Ce�H������G���=q�}p�CgxR                                    By��
  "          @���@��\@Mp��K���(�A�ff@��\@1G��c�
��G�A�G�                                    By��  "          @�(�@�z�@9���y������A�(�@�z�@�����(�A���                                    By��V  "          @�\)@��H@����33�H�Ạ�@��H?���ʏ\�S�A�z�                                    By��  
�          @�@\(�?�Q����
�o�RA���@\(�?O\)�����y�HAT                                      By��  T          @��
@��H?����Ǯ�]ffA���@��H?���˅�d=q@���                                    By�!H  
�          @��@!G�?u���H��A�(�@!G��#�
����\)C��R                                    By�/�  �          @��?�����\)���
¢C�e?����������C�1�                                    By�>�  
�          @���>�z�>����
=¬z�BZ��>�z�(����R©��C��                                     By�M:  
Z          @��>aG�=�Q�����°�=A���>aG��p�����R¥ǮC�}q                                    By�[�  �          @�녿s33����\)£��CM�f�s33������33� Cl�\                                    By�j�  
�          @�  �G���
=��\#�Cl\)�G��	����aHCy��                                    By�y,  
�          @��p�׾���\¤33CN5ÿp�׿�(���ff�{Cm^�                                    By���  �          @�G����\��{��R=qCq\)���\�3�
��z�#�Cz!H                                    By��x  
�          @�33�(Q��u��ff�O�HCk���(Q������(��5��Cp@                                     By��  H          @�33�0  ��
=���H�:�RCl޸�0  ������
=� (�Cp��                                    By���  �          @ٙ���z��L����
=�`Cv+���z��w�����C�
Cy�R                                    By��j  "          @�녿W
=�&ff���
=C|+��W
=�W���ff�f
=C��                                    By��  
�          @�녿h���7�����s�C|s3�h���e����R�U�CǮ                                    By�߶  T          @��ý�Q��33��ff�C��ý�Q��(Q���z��{��C��                                    By��\  �          @��
����=q����j  Ch{����Dz�����P  Cn�\                                    By��  �          @�  ��Q��=p���Q��I�Cl����Q��`���s�
�.  Cq�                                    By��  �          @�  ����O\)���R�h��Cq\���������ff�L(�Cv
=                                    By�N  �          @ٙ�����
=��z��m�Cu������<�������P\)CzJ=                                    By�(�  	�          @��H@���33@�G�B�W
C���@�����@�z�B���C�q�                                    By�7�  
�          @߮=���p�@ۅB��=C��
=����@޸RB�8RC�|)                                    By�F@  "          @�(�?����=q@�
=B�.C��{?��Ϳ�G�@�  B��\C�U�                                    By�T�  �          @�ff?��H���@ə�B�\)C�c�?��H���@ҏ\B�L�C���                                    By�c�  
(          @�  @���G�@У�B��HC���@�ÿ�=q@���B�33C��                                    By�r2  "          @��
@4z���@�(�Bw�C��=@4z῔z�@ۅB�{C�Ф                                    By���  
(          A?��@�\B��fC�B�?���  @���B�  C�E                                    By��~  
�          A	p�@\)���A�HB��C���@\)��Az�B��3C��                                    By��$  .          A��@"�\����A=qB��C���@"�\?E�AB�ǮA�(�                                    By���  
N          @��@0  >�@�\B��H@,(�@0  ?�
=@�\)B��A��                                    By��p  T          @�@3�
>��@��
B��)@C33@3�
?�@��B�#�A�
=                                    By��  T          @�33?��R<#�
@ȣ�B���>W
=?��R?h��@�ffB�(�A�
=                                    By�ؼ  �          @��@ ��=�\)@��B�
=?�\)@ ��?s33@�\)B�ǮA���                                    By��b  
Z          @�@*�H���@�ffB�C�.@*�H?@  @��B�G�Az=q                                    By��  
(          @�(�@Q�0��@��B��C��R@Q�>Ǯ@�\B��=A
=                                    By��  T          @�
=@=p�>\)@�G�B���@1G�@=p�?��H@�{B��A��                                    By�T  �          @��@
=?��
@�\)B�{A���@
=@
=@�  B���B'��                                    By�!�  �          Aff@&ff=��
@�B��?�\)@&ff?��H@�\B�� A�                                      By�0�  �          @�p���녾��
@��\B�ffCZW
���?W
=@�G�B�aHB��                                    By�?F  "          @�����+�@�Q�B�
=Ckٚ��>�@���B�8RCL�                                    By�M�  �          @�=q?��\�aG�@�{B��C��?��\?Y��@���B��)B�                                    By�\�  �          @�33?@  ��\)@��HB���C��
?@  ?(��@��B���B&33                                    By�k8  �          @����J=q>�\)@��
B~33C.޸�J=q?�p�@�  Bv=qC��                                    By�y�  .          @������H�#�
@�
=BG�HC4
���H?O\)@��BD�
C)�3                                    By���  
�          @�\)��=q��@��RBB\)C:xR��=q>�=q@�\)BCQ�C0�{                                    By��*  T          @�����{��G�@��HB:��C@���{�k�@�B?Q�C6                                    By���  
�          @�\)�����B�\@�p�B@�C=�����<�@�\)BC�RC3�H                                    By��v  �          @�p�����@  @���B7��C<�3���=#�
@�ffB:�C3��                                    By��  
�          @����þ�=q@�{B1��C6�R����?�@�p�B1�C.Y�                                    By���  	`          @����H<#�
@�{B;��C3�H���H?W
=@�(�B8�C*�{                                    By��h  "          @����33�\)@��\B>�C5����33?8Q�@���B<(�C,�                                    By��  
�          @���������H@�p�B9{C9s3���>�33@�B9��C0!H                                    By���  "          @�R���>�@�z�BB�C2k����?u@���B>��C(�                                     By�Z  "          @�ff���>���@�(�B5=qC0�����?���@�Q�B0\)C'Ǯ                                    By�   
�          A����\��z�@�=qB.�\C6�����\?
=@���B-��C.B�                                    By�)�  
�          A (���
=�\@��B1ffC7� ��
=?�\@��HB1  C.�3                                    By�8L  "          A�����H�Ǯ@��\B.��C7�
���H>��H@�=qB.Q�C/8R                                    By�F�  "          A �����?   @���BA�HC.�f���?�Q�@��
B;ffC$�
                                    By�U�  "          @����Ϳ#�
@�{BT�C<:�����>��R@��RBV
=C/�R                                    By�d>  T          A�����H���H@ٙ�BT33C9�=���H?
=@�G�BS�
C-aH                                    By�r�  T          A�����
�aG�@�G�BU\)C6�=���
?Tz�@ϮBS
=C*Y�                                    By���  T          A
=���H�fff@�BR=qC>}q���H>\)@ϮBU{C2Q�                                    By��0  "          @��b�\�\@��Bn33CKJ=�b�\��ff@�
=Bx
=C;O\                                    By���  
�          A (��z=q�c�
@�33Bn�C@�=�z=q>k�@���BqC0�3                                    By��|  �          @�Q��?\)���\@�
=B��CK�?\)��@��HB�G�C6s3                                    By��"  �          @���!녿�ff@�{B�ǮCOE�!녽�@陚B��qC6�\                                    By���  �          @��
�	����{@��B��CT8R�	����@���B�\C7�
                                    By��n  T          Ap��1G��
=@޸RBy��C\���1G�����@�  B��HCKp�                                    By��  
Z          AQ��zῥ�A\)B�.CV���z�>B�\A��B��C.B�                                    By���  "          A��:=q�&ffA��B��3C@���:=q?Y��Ap�B�{C#��                                    By�`  �          A�`�׿��\A{B�B�CD@ �`��>�ffA�HB�C,�f                                    By�  "          AG��y������A
ffB�k�CE)�y��>uA�B��3C0s3                                    By�"�  T          A����\)���R@�p�B_G�CA\)��\)=�Q�A Q�Bcp�C3�                                    By�1R  "          A=q���׿333A�\Bh��C;������?(��A�\BhC,z�                                    By�?�  "          A33��p��(��A	�By��C<}q��p�?G�A��ByQ�C*                                      By�N�  �          A(����Ϳ8Q�AffBm�
C<aH����?0��AffBm��C+�3                                    By�]D  
�          A����H���\A	��By{CD@ ���H>L��A
=B}�RC1ff                                    By�k�  "          A��u����HA  B�CL��u��W
=A�RB�p�C733                                    By�z�  �          A  �e���A�\B�L�CL�\�e��G�A��B���C5��                                    By��6  T          A���|�ͿaG�AffB�{C@�
�|��?#�
A�RB���C*�=                                    By���  .          A�y������A
=B�G�CGs3�y��>#�
A��B�B�C1�)                                    By���            A �����\��z�Az�Buz�CE!H���\=�G�A=qBz��C2��                                    By��(  "          A ����G��n{A(�Bs�\C?  ��G�?z�A��Bt��C-�                                    By���  �          A#33����(��A
=qBh��CQ�{�����\)A\)Bw�CA��                                    By��t  �          A#33�Z=q��AG�B�p�C<��Z=q?�
=AQ�B�u�C �                                    By��  T          A"=q�Q�>�p�A�HB��)C*#��Q�@�\A�B�k�C33                                    By���  �          A!���\��{A�
B�=qC<aH��\?�{AffB�#�CQ�                                    By��f  �          A ���#33��A�\B��C@���#33?�\)AB���CO\                                    By�  �          A33��Q�B�\A�B�(�CO� ��Q�?xQ�A��B��HC{                                    By��  T          A���  �   A��B���CF�{��  ?�p�Az�B��{C�                                     By�*X  �          A��   ��33A(�B�u�C=���   ?�\)A�RB�ǮC�\                                    By�8�  T          A���
=����AB���C>ٚ�
=?�ffAz�B��)C\)                                    By�G�  �          AG��z���Ap�B��3CDQ��z�?���A��B�p�C�)                                    By�VJ  "          Az�� �׾�A��B��=CAY�� ��?�p�A�B�\C��                                    By�d�  �          A�׿ٙ��L��A=qB���C:Ǯ�ٙ�?\AQ�B�C
G�                                    By�s�  T          A=q���
=�Q�A��B��C00����
?�A{B���B��\                                    By��<  �          A"�H��=�Q�A!B��qC/� ��?�33A�HB���B�Q�                                    By���  �          A$Q����>\A#�B���C�{����@p�A�
B���Bș�                                    By���  T          A$�׾���>�=qA$Q�B�8RC�)����@�A ��B��)BŸR                                    By��.  �          A%��\(�>uA$Q�B�\C$ff�\(�@ffA ��B��{B��                                    By���  �          A"�H���H>�ffA ��B��C"�\���H@�A��B�(�B�z�                                    By��z  �          A"ff��
=>�G�A�
B�Q�C%:��
=@��A  B���B��                                    By��   �          A Q���H>�=qA�B�33C,.���H@�
AB���C��                                    By���  "          A!p��ff=�Q�AB�  C1���ff?�33A�HB�k�C	�\                                    By��l  T          A$�׿�(�?   A!�B��C#�Ϳ�(�@
=A��B�  B�8R                                    By�  �          A&�R����?.{A%�B��RCzῙ��@%�A Q�B�B�B�\                                    By��  
�          A%녿(��?W
=A$Q�B�#�C n�(��@.�RA�HB�p�B�W
                                    By�#^  T          A#�
�#�
?z�HA#
=B�B�#׼#�
@7
=AG�B��\B�aH                                    By�2  
�          A#��:�H?.{A"�RB���C	��:�H@$z�AB��qBӨ�                                    By�@�  �          A$(���
=�8Q�A ��B�33C9^���
=?��HA�RB��)C
z�                                    By�OP  
f          A'�����<��
A%B�\C3:Ό��?�(�A"�HB���B���                                    By�]�  H          A&ff��
=>�  A$(�B�8RC+�=��
=@�A z�B�L�B�#�                                    By�l�  
�          A'�
��33>aG�A$��B���C-^���33@
�HA!G�B�\C@                                     By�{B  T          A'��p����A"�HB��qC7h��p�?��A Q�B�.C�                                    By���  
�          A'��(�þL��A"=qB��RC8n�(��?�p�A�
B��C�q                                    By���  �          A&=q�.�R��\)A (�B�aHC9���.�R?��A{B�G�C�                                    By��4  
4          A%���Fff�\A�B��fC:���Fff?\A(�B�
=C�
                                    By���            A$���1G���Q�A�HB��)C5��1G�?���A(�B��=C��                                    By�Ā  
�          A$���g
=��\)A��B���CEL��g
=?=p�Ap�B���C(c�                                    By��&  �          A%��\)��=qA�HB�
=CI���\)>�\)A��B��\C0
=                                    By���  "          A'��n�R���A\)B��=CC���n�R?^�RA�B�
=C&�                                    By��r  
�          A'\)��33��\)A�B�RCEn��33>�AffB��C-�q                                    By��            A��녿�  A	p�BqffCH�R��녽#�
A(�By��C4s3                                    By��  �          A#���Q���A\)Bi�CH����Q���A�\Brp�C5��                                    By�d  	.          A%G����H���HABl�\CD����H>��RA�Bq��C0��                                    By�+
  �          A!����=q���A��Bq��CB���=q>��A�Bup�C.aH                                    By�9�  �          A�
�z�H�#�
AffB�\C=33�z�H?�Q�A��B�z�C#�                                    By�HV  �          A%p���ff�}p�A�Bx�C?�f��ff?W
=A{By��C)�{                                    By�V�  �          A���{����A��Bl��CB�R��{>��A
=qBpC/:�                                    By�e�  
Z          A ����\)�aG�A
�\Bn�
C>���\)?^�RA
�\Bn�HC*
                                    By�tH  	�          A!����
��(�A��Bg{CF�����
=#�
A�Bn�C3�{                                    By���            A%G������ ��A
�\Bc�CH�q�����aG�A{Bl��C6k�                                    By���  T          A<  ��(�����A(�Bh�CC5���(�>��A{BlC0�                                    By��:  �          A0  ��p�@Q�A
=B��\C���p�@�33A��Bg\)CxR                                    By���  �          A)��R�\@�
=A�HBo�B���R�\@��A{BGB�                                    By���  J          A)G���  A@�\)B{B�zᾀ  A(�@�z�A�{B�\                                    By��,  
�          A)G���ff@���@�p�B*z�B��\��ffAG�@�A���B���                                    By���  "          A+�
��@��
@���B+33B�Ǯ��A
=@���A��B��                                    By��x  T          A/�
�8Q�@�{@�33B6\)B����8Q�A=q@�33B�HB�33                                    By��  T          A3�
>�A ��@��HB0��B�.>�A�
@�Q�B33B�8R                                    By��  T          A6=q?!G�A=q@��RB1�B�#�?!G�A�@ÅBffB��                                     By�j  �          A7\)?�(�A��@�Q�B*ffB�ff?�(�Az�@�33A��B���                                    By�$  �          A<Q�?��Az�@�
=B*�HB���?��A (�@�Q�A�z�B�Q�                                    By�2�  
�          AC
=?�z�A (�A=qBA�
B��{?�z�A�
@�Q�B��B�z�                                    By�A\  �          AMG�@(Q�@��A6=qByQ�Bp�@(Q�@�(�A (�BL�B���                                    By�P  �          AJ�R�xQ�A3�@��A��RB��f�xQ�A@  @��A3�
B�B�                                    By�^�  �          AU���=qA%���׮��\B�����=qA33�z��%B�
=                                    By�mN  �          AMp����\A�H�ۅ�ffB�#����\A  ����-�B�aH                                    By�{�  
Z          AA�i��A{�׮�33Bހ �i��@�\)�	��4Q�B�                                     By���  
�          AA��8��Aff��=q�p�B�L��8��A z����333Bۊ=                                    By��@  �          AF=q���A+���
=��ffB�  ���A
=�{�#B�{                                    By���  �          A=����G�A$Q����R��RBĽq��G�A�������$G�B�z�                                    By���  
�          A>�\��G�A"�\��  ��ffB��H��G�A
{� ���*=qB���                                    By��2  �          A=G��(��A!���H��=qB�k��(��A���{�-��B�                                    By���  T          A:ff�aG�A���=q����B�z�aG�A�R�G��/G�B��                                    By��~  
�          A9��?(�A!��������\)B�\?(�A
=q���R�&�B��H                                    By��$  �          A8Q�?B�\A#33�����p�B�� ?B�\A��������HB�(�                                    By���  �          A5G�?#�
A�R���
����B���?#�
A(������#�RB�p�                                    By�p  
(          A3�?}p�A{����޸RB��{?}p�A�
�����!�B��q                                    By�  �          A1p�?��RAQ����\��{B��\?��RA�R��33�Q�B�B�                                    By�+�  �          A0��?�\)A��������ffB��?�\)A�
��{��\B��                                    By�:b  @          A*�\?\(�A������\)B��?\(�A (���
=�#z�B�Ǯ                                    By�I  
�          A*�R?fffAp����\��G�B�?fff@������$��B�(�                                    By�W�  �          A)��?J=qA��������B�L�?J=q@�
=��{�#�\B��                                    By�fT  "          A(z�?s33A���=q���
B�W
?s33A (������ (�B�p�                                    By�t�  T          A#
=>L��A�������(�B�� >L��@�����=q� B��                                    By���  T          A!G���\A������G�B����\@�
=�θR�ffB��\                                    By��F  T          A z�
=A  ���
����B��=�
=@��������G�B��3                                    By���  
�          A ��?!G�A�
��p���33B���?!G�@�Q����H���B��3                                    By���            A"=q?ǮA{���
����B�{?Ǯ@�33��Q���B��
                                    By��8  �          A ��?\A�
���R���B�L�?\@�{��=q�#�B��                                    By���  �          A�?�  A
{��ff����B�\?�  @�\��G��#�
B�#�                                    By�ۄ  �          A!G�?��A33��\)��  B��?��@�z���33�#p�B��                                    By��*  �          A ��?���A
�\�����\)B��H?���@�33��33�$G�B��R                                    By���  T          A ��@�\Az���z���33B�
=@�\@���
=�(  B�33                                    By�v  �          A Q�?��RA	����R��(�B���?��R@陚��=q�#B�z�                                    By�  "          Aff?��HAz���z���33B�#�?��H@�\)��\)�#z�B��                                    By�$�  
�          A�R@33A	�������\)B�(�@33@�\�˅���B��3                                    By�3h  
�          AG�?�33A����{�ָRB�?�33@�G��ə����B��
                                    By�B  "          A�H@z�A33��\)��33B�� @z�@��\��RB��                                    By�P�  
�          A�@
�HA=q�����  B�@
�H@�\��{�$
=B��                                    By�_Z  
�          A�H@   A{��
=��
=B���@   @�G��љ��%�B��                                    By�n   �          A\)@%A  ����B�#�@%@ۅ��\)�*=qB���                                    By�|�  "          Aff@,��Ap����
�ޏ\B��@,��@�Q��θR�"B��H                                    ByL  "          A@333A�����أ�B��)@333@����ʏ\�B��{                                    By�  T          A�@@  AQ����R��33B�.@@  @�\)�ə���B��                                    By¨�  �          A�@C�
A�R��(��߮B��@C�
@ڏ\��{�"��B�Ǯ                                    By·>  �          A��@?\)A �������(�B�8R@?\)@�p���33�(33B���                                    By���  T          Aff@8��A������B���@8��@�
=����)\)B�u�                                    By�Ԋ  
�          A
=@@��Ap���p���B�33@@��@�p���
=�*(�B�u�                                    By��0  "          A�H@:=qA���p����B��@:=q@�ff��\)�*�B���                                    By���  "          A�H@=p�A=q�����z�B�
=@=p�@�
=���)�B��                                     By� |  �          A�R@@��A ����ff�B���@@��@��
��  �+��B�
=                                    By�"  
�          Aff@7
=Ap���ff��B�  @7
=@�����Q��,33B�aH                                    By��  �          A{@>{@�\)�����
=B�.@>{@�����G��-��B�\                                    By�,n  
�          Ap�@A�@�\)��z����B�k�@A�@�G���{�+z�B�Q�                                    By�;  T          AQ�@?\)@�{���H��B���@?\)@�Q���z��+33B���                                    By�I�  "          A  @0��A���(����B�p�@0��@�\)��\)�&��B�k�                                    By�X`  "          Aff@'�A�������B�33@'�@أ���33�$�B���                                    By�g  T          A�\@&ffA�����\��z�B�p�@&ff@�
=��{�'(�B��q                                    By�u�  �          A(�@%�@��
����  B��3@%�@θR��\)�+�B��                                     ByÄR  �          A=q@   @�
=��\)��B�.@   @���Ϯ�.��B���                                    ByÒ�  
�          A�
@{@�=q��Q�����B�  @{@�z�����.�B���                                    Byá�  
�          A@&ff@���������B��\@&ff@�{��(��.��B�{                                    ByðD  �          Ap�@ ��@����������B�Ǯ@ ��@�ff��(��/{B��                                     Byþ�  "          AG�@A  ��Q���G�B�L�@@�p���{�!(�B���                                    By�͐  �          A(�@�RA33��(��͙�B�L�@�R@�������(�B�k�                                    By��6  �          AQ�@{Az��~�R��  B��R@{@�Q���{���B��                                    By���  �          A  @ffA���b�\���B�aH@ff@�(����H�=qB�33                                    By���  
�          A\)@   A	�W
=��z�B��)@   @�\)��{�G�B�\                                    By�(            A�?�p�A	��XQ���33B�(�?�p�@�\)���R���B�aH                                    By��  �          A�\@�A���X�����B�L�@�@�z����R���B�W
                                    By�%t  �          A��@A	���c33���HB���@@����(��B�u�                                    By�4  �          A  @ffA	G��`  ���RB�� @ff@������\���B�Q�                                    By�B�  
�          A�@   A	��Y����  B���@   @�R��  ���B��                                    By�Qf  �          A?�p�A(��Vff��{B��q?�p�@�33�����B���                                    By�`  
          A33?���A�׿�
=�AB�L�?���A   ��(����
B�                                      By�n�  
�          A��?���A
�R��(��/�B��{?���@�p��z=q�̸RB�L�                                    By�}X  T          A�?�A\)��Q��G�B��?�@��
�R�\���B�Ǯ                                    Byċ�  "          A�H?�(�A33�^�R����B�W
?�(�@�ff�>{��33B���                                    ByĚ�  �          A
=q?��Ap���Q�����B���?��@���Tz����B�                                    ByĩJ  �          A  ?�Q�A����
��
=B�ff?�Q�@���I����B��\                                    Byķ�  �          A�\?��A{�����=qB�33?��@��O\)��G�B�W
                                    By�Ɩ  �          A�?���A=q��  �"�RB�u�?���@�R�e��(�B�8R                                    By��<  �          A�?�=qA=q����"�\B�W
?�=q@�{�l���ȣ�B�(�                                    By���  �          A
=@ffA���x��B���@ff@�ff��z���RB��                                    By��  �          A@   A\)�B�\��B��@   @��
����ffB�=q                                    By�.  �          A��@%�@�\)�Z�H��\)B��@%�@��������B�G�                                    By��  T          A�@#�
@���[���  B��@#�
@�����{�p�B��R                                    By�z  �          A�\@8Q�A ���S33��
=B��\@8Q�@��
���\�ffB�u�                                    By�-   �          A�\@3�
@���p����=qB���@3�
@����\)�{B���                                    By�;�  �          A�
@9��@�p��p����ffB���@9��@��
��  ��B��R                                    By�Jl  "          A\)@Fff@�z����H��33B���@Fff@�Q���  �"�B~�\                                    By�Y  T          A�@Fff@���������
B��H@Fff@�  �����#(�B~�\                                    By�g�  
�          A  @W�@������ՙ�B�  @W�@��������#33Bu=q                                    By�v^  T          A@Y��@�����H��
=B���@Y��@�33��\)�&�Bs�                                    ByŅ  �          A{@a�@�������=qB��)@a�@�33��{�%33Bo��                                    Byœ�  �          A�\@XQ�@���Q���p�B�p�@XQ�@�
=��{�$�Bu�                                    ByŢP  y          A=q@Z=q@�33��=q��p�B���@Z=q@�(��Ǯ�&�\Bsz�                                    ByŰ�  �          A�@W�@�z���  ���
B�Q�@W�@����%�Bup�                                    Byſ�  �          A@\(�@��������Q�B��)@\(�@�G��ȣ��({BqG�                                    By��B  �          Az�@g�@�{�����=qB}�@g�@�z�����.�
BeG�                                    By���  �          A(�@\��@��
��{����B��)@\��@���ə��+G�Bn{                                    By��  /          A{@K�@�{�����G�B��@K�@�
=��(��(�RBw�
                                    By��4  y          A��@C33@�Q��~�R��33B���@C33@Å����#p�B}��                                    By��  "          A\)@%@��P  ��z�B�Ǯ@%@ָR���\��B�                                    By��  �          A�
@   A ���J=q����B�p�@   @��H������RB��
                                    By�&&  y          AG�@"�\AG��P����ffB��@"�\@�33��z���B�Q�                                    By�4�  "          A��@=qA��B�\���B�W
@=q@ᙚ��
=�
=B�=q                                    By�Cr  "          A��@�
A��0  ��
=B��@�
@�
=��
=�{B�p�                                    By�R  "          Az�@{AG��*�H���
B��@{@������
B��)                                    By�`�  "          A  @
�HA��*=q��p�B�Ǯ@
�H@�\)������HB���                                    By�od  �          A��@��A{�#�
���B��{@��@�=q���\�(�B��                                    By�~
  y          Az�?�(�A�
���s33B��?�(�@�
=������B��                                     Byƌ�  �          A�
?�A���
�m�B��?�@�
=���
��p�B�=q                                    ByƛV  �          Ap�?���A��
=q�bffB�?���@������R��(�B��H                                    ByƩ�  �          Az�?�33A�H��ff�=��B���?�33@�=q��z����HB�W
                                    ByƸ�  �          A��?�(�A\)�����K�
B���?�(�@�G���G���{B�G�                                    By��H  �          A�\?�Q�A��
�H�a�B�k�?�Q�@���Q���B��\                                    By���  �          A{?���Ap����~ffB��?���@�G���\)� ��B�u�                                    By��  �          A?�=qA(��У��)��B��?�=q@������י�B�#�                                    By��:  �          A��?�(�A(���{�=qB�.?�(�@�  �p���ʣ�B�                                    By��  "          A(�?�A�
�������B��3?�@�\)�q���=qB��\                                    By��  T          A
�R?��A
=������Q�B�  ?��@���a���z�B�{                                    By�,  T          A
�\?ǮA{�\�!B���?Ǯ@�\�x���ծB���                                    By�-�  "          A
�R?�z�A�R���R�ffB��?�z�@��
�xQ�����B���                                    By�<x  �          A��?�{A �ÿ����RB��H?�{@陚�l(���G�B��                                    By�K  �          A33?�(�@�������y�B�(�?�(�@����z���B���                                    By�Y�  �          A�H?�z�@�ff�\)���
B���?�z�@�(���z��
\)B�{                                    By�hj  �          A�R?���@�=q��
=�[\)B�Ǯ?���@���������z�B�L�                                    By�w  
�          A��?�\)@�
=�G��g33B��f?�\)@������R��z�B�.                                    Byǅ�  /          A�R?�(�@��׿��H�_
=B���?�(�@��H��p���ffB�Ǯ                                    Byǔ\  y          AG�?��
@���aG����
B��?��
@�G�����,  B��f                                    Byǣ  a          A33?Q�@ᙚ��33��{B���?Q�@�Q���Q��<�\B���                                    ByǱ�  �          A?��@���#�
��p�B�Ǯ?��@����\)�B�aH                                    By��N            A��?O\)@��H�!G����B���?O\)@�  ��p���B�                                    By���  �          A�?���@�33��
��p�B��
?���@����\)�{B��R                                    By�ݚ  �          A ��?Ǯ@�(����]B��
?Ǯ@ָR���
��33B��q                                    By��@  �          @��R?˅@��
��\)�;�B�k�?˅@�G��u���ffB���                                    By���  a          @��R?��R@�ff���
�z�B�{?��R@޸R�b�\�ә�B���                                    By�	�  
�          @��?W
=@��
�����B�Ǯ?W
=@�{�W��ǅB���                                    By�2  �          A (�?s33@��
��z��B�=q?s33@����_\)�Ώ\B��H                                    By�&�  �          A z�?��
@�  �����B��?��
@޸R�j�H��{B�=q                                    By�5~  �          @�
=?�\)@�p��У��=G�B��R?�\)@�=q�x���陚B�=q                                    By�D$  �          @��?�
=@�=q�����V=qB�  ?�
=@��������
=B��                                    By�R�  /          @�{?�p�@�\��=q�7�B�B�?�p�@׮�tz���  B�(�                                    By�ap            A��@ff@����!G���B�Q�@ff@�����ff�
\)B���                                    By�p  �          A�?�\)@�G��  �z�RB�\?�\)@�\)�������B���                                    By�~�  �          A=q?��\A (�����z{B��?��\@�����\)B�G�                                    Byȍb  T          A�?�33@���(��s
=B��\?�33@���G���\B�{                                    ByȜ  �          A��?���@���	���pQ�B��?���@ۅ������B�G�                                    ByȪ�  �          A  @
�H@��H���R�
ffB�\@
�H@�\�fff��p�B��f                                    ByȹT  �          A=q@=q@�  �+���ffB�z�@=q@���A�����B���                                    By���  �          A{?���@���8Q���=qB�33?���@�  �G�����B���                                    By�֠  �          Aff@�@��׿(����(�B�\@�@��B�\��33B�u�                                    By��F  �          Ap�?�Q�@�녿J=q���B�(�?�Q�@��K����B���                                    By���  "          A
=?˅@��Ϳ����33B�8R?˅@�=q�qG����B���                                    By��  T          A
=?�Q�@�\)�1�����B��=?�Q�@Ǯ��\)��\B��                                    By�8  T          A�?!G�AG�����r�\B�\)?!G�@�\)�HQ���  B��3                                    By��  T          A��?(��A����j=qB���?(��@�z��E���33B��                                    By�.�  T          A?���A=q��{�0��B�{?���@�
=���\���
B�8R                                    By�=*  T          A{?���@�\�$z���  B��
?���@�(����\�33B�(�                                    By�K�  �          A ��?�G�@��U���B�8R?�G�@�{����)
=B���                                    By�Zv  
�          A Q�?���@��L�����
B��
?���@��������%��B��                                    By�i  
�          A   ?�\)@�p��I�����
B��{?�\)@��������$�RB��
                                    By�w�  
(          @��?�33@����!G����RB�aH?�33@�
=����z�B��                                    ByɆh  "          @���?��@�\)�
�H��B�.?��@�����{��B��                                    Byɕ  
�          @�33?�p�@�����^=qB�L�?�p�@�  ��p�� �HB��                                    Byɣ�  �          @���?��H@��˅�<z�B���?��H@ҏ\�y����Q�B�
=                                    ByɲZ  "          @���?���@��ÿ����>ffB��H?���@��
�{����HB��\                                    By��   
�          @�=q?��R@�׿�G��Pz�B�L�?��R@љ����\���
B���                                    By�Ϧ  "          @��\?�p�@�׿�ff�T��B�� ?�p�@�G���(���ffB���                                    By��L  T          @���?��@�{�����[�
B���?��@�ff����� �
B��{                                    By���  
�          @�=q?��@�\)�����Z�\B�z�?��@Ϯ���� B��=                                    By���  �          @��?��
@����=q���HB�(�?��
@�
=��p���B�8R                                    By�
>  "          @�(�?��@�
=��\���HB��?��@�=q���\�G�B�\                                    By��  "          @��
?�\)@�G����H�f�\B�\?�\)@Ϯ������B�aH                                    By�'�  T          @�33?�33@��ÿ�
=�F=qB�  ?�33@�=q��G���(�B�                                    By�60  
�          @�p�?�ff@��H��{�;�B��)?�ff@����\)���B��3                                    By�D�  T          @�(�?���@�p������G�B�W
?���@�������z�B��                                    By�S|  "          @��?��@�\)�Q��{�B��H?��@��
���R�
p�B��                                    By�b"  �          @�p�?z�@����R��
=B�33?z�@�����H��
B���                                    By�p�  
�          @��R?#�
@�(��8������B�#�?#�
@�������!�RB��                                    By�n  �          @��
?J=q@��$z���\)B���?J=q@Å��33�
=B�Q�                                    Byʎ  �          @�(�<#�
@�녿n{�أ�B��f<#�
@�=q�\����(�B��H                                    Byʜ�  "          @��R?s33@�R��
=�Ip�B�z�?s33@Ϯ�������B�Q�                                    Byʫ`  �          @�?�{@�����u�B���?�{@�\)���\�	��B��R                                    Byʺ  "          @��
?�  @�R����	p�B��R?�  @�p��c33��(�B��                                    By�Ȭ  a          @���?�  @�{���^�RB��3?�  @��
�9������B�p�                                    By��R  G          @���?^�R@�
=���c�
B�Ǯ?^�R@�Q��&ff���B��f                                    By���  �          @��
>u@�=q�&ff��p�B�p�>u@�p��H����
=B��                                    By���  
�          @���\)@�=q��R��{B�L;�\)@��G�����B��3                                    By�D  
M          @�>�ff@�
==u>�B�
=>�ff@���(���Q�B���                                    By��  T          @�
=?��@�ff?��A>�RB��?��@陚�u��B�.                                    By� �  T          @�?�=q@�  ?�\)Ad  B�=q?�=q@�R�+���=qB���                                    By�/6  "          @�{?�p�@�G�?�(�Av=qB�.?�p�@�G����\)B��
                                    By�=�  
�          @�R?fff@���?��RAz�RB�k�?fff@�G��   �xQ�B��                                    By�L�  �          @���@�@�@p��A癚B���@�@�=q?�{A!�B��f                                    By�[(  T          @�Q�?p��@�G�@�B*(�B��?p��@���@6ffA�  B��                                    By�i�  T          @�
=��@u�@�ffBp�B��)��@��
@�  B!�RB��                                    By�xt  
          @�  =�Q�@���@�{Ba33B�� =�Q�@�\)@�33B��B�G�                                    Byˇ  G          @�G�>�  @���@��BX��B��q>�  @�@�z�B	(�B��\                                    By˕�  �          @��>\)@��@�=qBL�B�Ǯ>\)@�@���A���B���                                    Byˤf  �          @��\=��
@��
@��
BM�HB�\=��
@��@��HA�(�B��{                                    By˳  
�          @��ͽ�G�@�G�@�G�BR��B��\��G�@���@���B  B��
                                    By���  �          @��H���R@�  @�p�B[��B��쾞�R@�p�@��B�B�z�                                    By��X  "          @����(�@�{@�z�B[�
BÙ��(�@˅@�\)B�B��                                    By���  T          @��þ�p�@\)@�p�Bk�HB�����p�@�G�@�z�B�RB�\                                    By���  �          @�G���@`��@�B{��B�p���@�@���B+�B��{                                    By��J  "          @�����G�@U@���B��)B�  ��G�@��@�ffB1p�B�
=                                    By�
�  �          @�ff��\@ ��@�  B�#�B���\@��@�{BJG�B�                                      By��  
�          @��R�+�@33@�\B��)B�\)�+�@�{@\BP\)B�.                                    By�(<  �          @�Q��@#33@�Q�B�\)B�(���@��@�p�BH��B�=q                                    By�6�  T          @����@U@�Q�B���B���@���@���B4
=B�aH                                    By�E�  �          A �׾W
=@@  @�ffB�{B�\�W
=@���@�{B>�B�z�                                    By�T.  
(          A ��>�=q@<(�@�\)B��)B�u�>�=q@��@�\)B@ffB�33                                    By�b�  �          @�{>\@p�@���B�aHB��\>\@�{@�ffBM��B�=q                                    By�qz  T          @��R>�(�@%@�Q�B�u�B�  >�(�@���@�(�BI�B�33                                    By̀   �          @�p�>�=q@5�@�z�B��B��f>�=q@�\)@�BB(�B�                                      By̎�  
�          @���#�
@7�@�33B�aHB��\�#�
@�Q�@�(�B@�RB��
                                    By̝l  "          @�p���p�@\)@�  B��HB��
��p�@�
=@���BK��B��                                     By̬  
�          @��8Q�?��
@�{B��B�(��8Q�@��
@�=qB`
=B�Ǯ                                    By̺�  �          A Q�.{@�@�\B�L�B��þ.{@�ff@�\)BN33B��                                    By��^  
�          A (�=�\)@#�
@�\B���B���=�\)@�=q@�{BJ��B�z�                                    By��  
�          A��>��@�R@��RB�#�B�(�>��@���@��HBM�RB�(�                                    By��  
�          A�>.{@*�H@��B�u�B��=>.{@��R@�
=BH�B�(�                                    By��P  T          A=q>�@2�\@�(�B�W
B�33>�@��@���BD33B�\                                    By��  �          A33>��R@N�R@��B�B�B��>��R@�{@�(�B7�\B���                                    By��  �          Ap�=�@<��@���B��B�� =�@�{@�\)B>��B��\                                    By�!B  �          A��>8Q�@   @��HB�33B��>8Q�@��@ӅB[�B��\                                    By�/�  T          A ��?��?h��@��HB�=qA���?��@i��@�  Bq��B��                                    By�>�  
�          @��?У�?5@���B�=qA�(�?У�@\(�@��Bu��B�G�                                    By�M4  
�          @�\)@�\?E�@�p�B�8RA��@�\@^{@�z�Bn�
Bm�                                    By�[�  G          @�\)@p�?��\@��B�ǮA�@p�@x��@�33B_z�Bq�                                    By�j�  �          A (�@��?��R@�B��3A�ff@��@��\@θRBW�Bn{                                    By�y&  y          A ��@G�?�Q�@���B��qB��@G�@�G�@��BS��Bx\)                                    By͇�  �          A�@!�?��
@��
B���B{@!�@���@�
=BO�\Bp=q                                    By͖r  �          A�
@�R@�
@�z�B���B*��@�R@�@�z�BKB���                                    Byͥ  �          A��@�@\)@�z�B��B1��@�@�33@��BF��B��\                                    Byͳ�  "          A��?�z�@��@��B��B_�R?�z�@��H@ʏ\BG(�B��)                                    By��d  �          A�H?W
=@G�@�Q�B�p�B��q?W
=@�(�@��
B7��B��                                    By��
  T          A ��>�33@Mp�@��
B��RB��>�33@��@��RB4�B��                                    By�߰  �          A Q�>.{@L(�@�B�B��H>.{@���@�{B4��B��=                                    By��V  �          A>Ǯ@E�@�\)B���B�p�>Ǯ@��H@�33B8��B���                                    By���  "          A{>���@J�H@�
=B��3B���>���@�p�@��B6\)B��                                    By��  �          A�H>���@c33@�33B�33B�L�>���@��@���B+(�B�u�                                    By�H  "          A>#�
@hQ�@�B}�\B�
=>#�
@���@�p�B'�B�                                      By�(�  �          A ��>B�\@}p�@�Q�Br{B��{>B�\@�  @��\BQ�B��\                                    By�7�  �          A ��>W
=@tz�@��HBv��B��{>W
=@���@�ffB B�                                      By�F:  T          A (��k�@y��@߮Bs(�B��k�@�{@�=qB(�B�B�                                    By�T�  T          @���ff@dz�@��B|
=B�Q��ff@��@�Q�B&=qB���                                    By�c�  a          @�p����@i��@�  By
=B¸R���@�
=@�p�B#�B�
=                                    By�r,  G          @����z�@|��@�Q�Bj�B��)��z�@��@��\Bz�B�ff                                    By΀�  T          @��Ϳ�p�@l��@ۅBq��Bؽq��p�@�
=@�Q�B��B�B�                                    ByΏx  a          @�����@J�H@��B�B����@�33@�\)B/�\B��                                    ByΞ            @�zῌ��@>{@�\)B���B�uÿ���@�{@��
B5p�B���                                    Byά�  �          @���u@.�R@�G�B���B��Ϳu@��@�Q�B<�B���                                    Byλj  �          @�=q���@6ff@�ffB�
=BݸR���@�=q@�(�B8�B��                                    By��  
�          @��
��33@%@�\B��RB��
��33@�(�@��HB@�B�B�                                   By�ض  "          @���u@7�@陚B��B�ff�u@�(�@��RB:�RB�                                   By��\  �          @��\>�G�@A�@�{B��B�u�>�G�@��@���B4�B��
                                    By��  "          @���>Ǯ@R�\@��B�B�� >Ǯ@�@�=qB,G�B�(�                                    By��  "          @���?�@K�@��B���B��{?�@��\@��B.B�G�                                    By�N  
�          @�{?
=@S33@�ffB��B��?
=@�  @�ffB-33B�B�                                    By�!�  "          @���?\)@`��@��B}
=B��
?\)@���@�\)B%B��                                    By�0�  
�          @�{?z�@dz�@�=qB{��B�aH?z�@��R@�
=B$G�B�Ǯ                                    By�?@  T          @�?�R@_\)@�33B}�HB��)?�R@���@���B&�B�                                      By�M�  �          @�ff?#�
@Vff@�B�.B�=q?#�
@���@�z�B*��B�W
                                    By�\�  
�          @�
=>�p�@Mp�@���B�.B�  >�p�@��R@�G�B0=qB��                                    By�k2  
�          AG�>L��@@��@�B�(�B�p�>L��@��@��B7�B��                                    By�y�  "          A��?�@HQ�@�{B���B�Q�?�@��R@��RB3��B��                                    Byψ~  �          A ��>�(�@S33@�\B�33B�B�>�(�@�=q@�G�B.
=B���                                    Byϗ$  "          @�\)?��@g�@��HBz��B��3?��@�G�@�{B"\)B��\                                    Byϥ�  "          @�z�?�R@^�R@ᙚB}�\B��H?�R@�z�@��RB%\)B�
=                                    Byϴp  "          @�p�?!G�@e@���Bzp�B�33?!G�@�\)@�z�B"{B�\                                    By��  
�          A Q�?W
=@U�@�  B�L�B���?W
=@��\@�{B*�HB��                                    By�Ѽ  a          Aff?\)@W�@�z�B�p�B�?\)@�p�@��B,=qB�#�                                    By��b  G          A Q�?���@fff@��Bu�B�?���@�  @��
B{B��R                                    By��  �          @�z�?z�H@J�H@�\B���B�z�?z�H@�(�@��HB,G�B�(�                                    By���  �          @��>�33@QG�@��HB���B��>�33@�(�@��\B'�HB��
                                    By�T  �          @���?�z�@@  @�p�B�{B���?�z�@�p�@�  B-B��
                                    By��  
�          @�  ?�{@j=q@�ffBq{B�k�?�{@�ff@�G�B�B��                                    By�)�  �          @��?�@^�R@�p�B{B�p�?�@��R@�G�B%�RB��                                    By�8F  �          @�Q�?�  @1�@�=qB�ffB��?�  @���@�
=B4��B�Q�                                    By�F�  T          @�G�>���@{@�B�ffB�� >���@��@�33BC�B���                                    By�U�  T          @�33?O\)@9��@�B��
B��R?O\)@��R@�=qB5{B�{                                    By�d8  �          @�\)?�  @_\)@�=qBy\)B��?�  @�{@�B"  B�33                                    By�r�  �          A z�@��@e@�\)BeffBa33@��@���@��\B�RB��H                                    ByЁ�  �          @��@5�@b�\@��HB^  BM��@5�@��@�
=B�\B��                                    ByА*  �          A z�@J=q@a�@�\)BWQ�B@�
@J=q@�Q�@��B\)Bu{                                    ByО�  �          Aff@mp�@Z�H@��BMz�B*�
@mp�@��H@�  BffBa�                                    ByЭv  �          A�H@5@w
=@љ�BW33BV��@5@��H@���B=qB�                                      Byм  �          A�R?�G�@Vff@�RB|  B��=?�G�@�(�@�33B%�\B�=q                                    By���  T          @��>k�@s33@��
Bt(�B��>k�@��@��B
=B���                                    By��h  �          AQ콏\)@l��@��B|��B�Q콏\)@�  @���B!G�B�aH                                    By��  T          A\)>�?�\A�B��
B��R>�@��\@�z�B[ffB���                                    By���  �          A�R��33@!�A (�B���BÞ���33@�{@�z�BF{B�L�                                    By�Z  T          A�
    @=p�@�p�B�ǮB�      @�
=@��B7�B�                                      By�   
�          A  ��
=@)��@���B��fB���
=@�\)@�(�B@�\B�Ǯ                                    By�"�  �          A�
��\@{@��B���B�\��\@�(�@��BLB�p�                                    By�1L  T          A���=q@K�@�p�B�W
B�aH��=q@��\@��HB-\)B��                                    By�?�  �          A���.{@�
@�
=B�aHB��.{@���@ƸRBHffB��                                    By�N�  �          A\)�L��?�  Ap�B�{C �ͿL��@��\@ᙚBm�
B�
=                                    By�]>  T          A (��E�@J=q@ָRB�G�B�Q�E�@�G�@�B%B���                                    By�k�  �          @���(��@�G�@�p�B33B���(��@�z�?�
=Ai�B�=q                                    By�z�  T          @�\)���@mp�@��
Bb{B�����@��
@n{BB�W
                                    Byщ0  �          @�\)��ff?���@�B��3B�녾�ff@��H@��
BS�B�=q                                    Byї�  /          @�(�����@!�@�ffB�.B�\����@�Q�@��B@{B�G�                                    ByѦ|  y          @�z�G�@.{@��HB�ǮB��
�G�@�z�@�p�B8�\B�ff                                    Byѵ"  T          @�{����?�\@�
=B���B�
=����@�p�@���BWQ�B�
=                                    By���  �          A zᾳ33?�{@�G�B��RB�zᾳ33@���@�BU{B�p�                                    By��n  "          A�\�#�
@!G�@��RB��qB�W
�#�
@��@��HBAB��=                                    By��  �          A�}p�@.{@��HB�z�B��H�}p�@�33@��
B<�\B�                                      By��  �          A�H�^�R@.�R@�p�B�33B�ff�^�R@�z�@�{B=G�Bŏ\                                    By��`  "          A
=�G�?��RAffB��B��G�@���@�ffBR��Bŀ                                     By�  "          A�H�k�@G�A{B�z�B�Ǯ�k�@��\@��BP��B�p�                                    By��  �          Aff�fff?��HA\)B�k�B���fff@��H@�ffB`33B�B�                                    By�*R  �          A
�H�Q�@(�Az�B�k�B�\�Q�@���@�(�BG\)B��
                                    By�8�  "          A  �W
=@J=qA��B�{Bѽq�W
=@��
@�B4
=BÙ�                                    By�G�  �          A=q�aG�@B�\Az�B�.B�B��aG�@��H@�z�B8ffB�p�                                    By�VD  �          A�aG�@&ffA
�HB��HB�33�aG�@��H@�p�BF
=B�
=                                    By�d�  �          A\)�xQ�@P  A��B��qB���xQ�@�(�@�G�B5z�B�33                                    By�s�  �          A=q��  @~�RA�\B}33B�Q쿀  @��@�z�B �RBĊ=                                    By҂6  T          A  �fff@���AQ�B}�HB�B��fff@�  @�
=B ��B³3                                    ByҐ�  T          AQ���@z=qA=qB��=B�p����@�@�(�B%ffB��H                                    Byҟ�  �          A��:�H@xQ�A��B�W
B�.�:�H@���@�33B%33B�                                    ByҮ(  �          A�R��p�@�z�@�
=Bf33B��=��p�@�=q@��B�B��                                    ByҼ�  �          A\)��
=@��A Q�Br�RB��3��
=@�\@���BffB��\                                    By��t  	m          A��R@�Q�A (�Bv�BĀ ��R@��
@�(�B�HB��                                    By��  
          A=q�+�@�  A ��Bw�B��Ϳ+�@��
@��Bz�B��                                    By���  �          A�\����@}p�A�\B{\)B�zῬ��@��@�(�Bp�B�                                    By��f  
�          A����@|��A�B|�B��쿧�@�p�@�{B �\Bɀ                                     By�  "          A��p�@��
@�z�Be�B�p���p�@���@��B��B���                                    By��  "          A�R��@�  @�ffBf\)B���@�\@��RB��B֏\                                    By�#X  �          AG���(�@���@���Bgz�B܀ ��(�@�{@�33B
=B���                                    By�1�  "          A33����@��@�z�Bbz�B������@�(�@�  B(�B��H                                    By�@�  "          A
=�33@��
@�
=Bfz�B��f�33@�{@�p�BQ�B���                                    By�OJ  	�          A����@�ff@���BbQ�B�\)���@�@�=qB	  BՏ\                                    By�]�  
�          A�
���@�=q@�\)BY��B��)���@�Q�@���A�
=Bә�                                    By�l�  �          A�׿�@�p�@�BSB�\��A z�@�Q�A��
B��f                                    By�{<  
�          Az��ff@�Q�@�(�BJ��B֣׿�ffA(�@��Aܣ�B̊=                                    ByӉ�  �          AG���Q�@�{@�\BG�B��Ϳ�Q�Aff@���A���B�                                    ByӘ�  �          A �׿(��@�  @���B>
=B�=q�(��A�R@���A��B�G�                                    Byӧ.  T          A!�����@�ff@�  B0�RB��\����A�R@^{A��B�.                                    Byӵ�  T          A!��=u@�p�@�33B*B�#�=uA��@O\)A��B�Q�                                    By��z  
�          A�\=L��@�@�33BQ�B�8R=L��Az�@,��A}G�B�aH                                    By��   �          A �ÿ�RA ��@�
=B�B��Ϳ�RA��@�AH(�B�B�                                    By���  �          A!���O\)A(�@�Q�B(�B�8R�O\)A=q?�33A-G�B�aH                                    By��l  �          A=q?�z�A��@���A�Q�B�#�?�z�AG�?�=q@��B�8R                                    By��  �          A�@
=A�\@�z�A�
=B��)@
=A�?z�@Z�HB��f                                    By��  �          A�@�A{@��RA���B�(�@�A�?&ff@w
=B�G�                                    By�^  y          A��?�33A33@�G�A�B��R?�33A
=>�Q�@�B��{                                    By�+  T          A(�@  AG�@�{A��B���@  A�R?&ff@tz�B�(�                                    By�9�  T          A��@�H@�33@�  B�B��@�HAG�?�
=A"�\B�.                                    By�HP  �          A  @��@���@�{B�
B�.@��A�?�
=A;�
B�(�                                    By�V�  �          A��@  @��
@��
BffB���@  Az�@
=AL(�B���                                    By�e�  "          A�\@\)@��
@�{A�=qB��)@\)A�
?Q�@�G�B�
=                                    By�tB  �          A{@AG�@�z�@��A�
=B��@AG�A��?s33@�33B�8R                                    ByԂ�  T          Ap�@%A�\@p  A��B���@%A�=�\)>�ffB��)                                    Byԑ�  
�          A�?�Q�@�{@�G�A�=qB��?�Q�A?^�R@��B��                                    ByԠ4  y          Aff?�p�@�(�@���BB��{?�p�A(�@G�AF�RB�8R                                    ByԮ�  
Z          AG�?��H@�\@���B  B��H?��HAp�@�Ak�
B��                                    ByԽ�  �          A?�z�@У�@��
B&=qB��
?�z�A@*�HA�=qB�8R                                    By��&  S          A
ff?Y��@�33@�Q�BffB�G�?Y��A��@33Av�RB�=q                                    By���  	�          A
�\?(�@�=q@��\B B�W
?(�A��@�A~�RB��=                                    By��r  "          A
�H?�G�@�(�@�ffB�RB�z�?�G�A��@�RAnffB��3                                    By��  T          A	�?(��@��H@�p�B&��B�\?(��A@#33A�G�B���                                    By��  �          A	?�ff@�=q@�{B.��B�?�ff@�\)@9��A��B�p�                                    By�d  
�          A�?�G�@�Q�@�B>  B�ff?�G�@���@^�RA��
B��                                    By�$
  x          A
{?��
@��@��B=\)B��?��
@�
=@[�A�(�B��{                                    By�2�  "          A�H?��\@��
@VffA�p�B�{?��\AG���G��8Q�B���                                    By�AV  �          A�?�ff@�  @^�RAŅB��?�ffAQ�=u>�G�B��\                                    By�O�  x          A��?���@�\)@�33A���B��\?���Aff?���@��B�33                                    By�^�  �          @��H>���@�G�@��B=ffB��>���@ۅ@:=qA�G�B�Q�                                    By�mH  
�          @�G�>�=q@��H@�{BH=qB���>�=q@�33@;�A̸RB�                                      By�{�            @��
��(���(�@}p�B��RCz����(�?�@��B�{C0�                                    ByՊ�  �          @�=q��(���{?�{A9�C��쿼(�����@z=qBC~z�                                    Byՙ:  �          @������߮?+�@��HC��������\@`��A�\)C��H                                    Byէ�  "          @�녿����љ�?�z�A?�
C��3�������
@�Q�BQ�C��R                                    Byն�  "          @���\)�أ�?�AxQ�C�B���\)���@��B  C��                                    By��,  
�          @�Q쿕���@�A�p�C������z�@�p�B*�
C�f                                    By���  T          @�33�xQ�����@��A��RC�j=�xQ����R@��B8�C�ٚ                                    By��x  
(          @��Ϳs33��ff@^{B
(�C��3�s33�C33@��RBi�C|�                                     By��  �          @�z�����{@\)A���C�8R������@��B7ffC�G�                                    By���  "          @��H�=p�����@,(�A�33C��3�=p��\(�@���BP�RC���                                    By�j  
�          @�G����H��Q�@z�A�z�C�����H�HQ�@\)BN{C���                                    By�  "          @��H���
���R@��B<�RC|녿��
�
�H@�p�B�\)Cj�                                    By�+�  �          @������g�@ȣ�BiCy𤿨�ÿ+�@�ffB�ǮCNٚ                                    By�:\  
�          @�zῡG��S�
@�\)BuG�Cy.��G�����@��B���CA�                                    By�I  
�          @�z����{@��
B���Cg.���?&ff@�
=B�k�C �                                    By�W�  T          @�33���H?���@�\)B�{B�#׿��H@��\@�(�BQ=qBр                                     By�fN  �          @�
=�p��@c33@�Q�ByG�B��p��@��@�(�B=qB�k�                                    By�t�  �          @��þB�\?J=q@�RB��\B�\)�B�\@u@�{Bl��B��                                    Byփ�  �          @�G�=����'�@��B�\C�!H=���?\)@�
=B�B�B��                                    By֒@  
�          @�Q�<��
���@�G�B�#�C�` <��
?�z�@�Q�B�Q�B���                                    By֠�  
�          @��R���Ϳu@�z�B��\C��q����@�@��HB�ǮB�                                      By֯�  
�          @�zᾊ=q��\@��B���C��׾�=q?���@�ffB�.B�Q�                                    By־2  �          @���#�
��@�(�B���C��{�#�
@-p�@���B�ffB���                                    By���  �          @���Q�k�@�\B��=CT0���Q�@>{@�
=B��B��f                                    By��~  �          @�(���(���p�@��B�ffCz�{��(�@�\@��HB�=qB˔{                                    By��$  F          @��
��(��Ǯ@�{B��=C~�׾�(�?��H@���B�.B�                                    By���  
�          @�G��L�Ϳn{@��RB��C�녾L��@��@�(�B��B��\                                    By�p  �          A ��>\)�uA ��B���C�(�>\)@Tz�@�=qB�#�B��                                    By�  T          A Q���?�\A   B�\)B�{���@u�@�G�Bu��B�k�                                    By�$�  "          @��R>W
=?�ff@��\B��HB�
=>W
=@�\)@��B^��B��q                                    By�3b  T          @��?
=?�\)@�  B���B���?
=@�ff@ǮBMp�B�L�                                    By�B  T          A��?L��@  @��B��B��)?L��@���@�=qBB=qB���                                    By�P�  
�          A(�?�{@ffA��B�p�B��\?�{@�G�@˅BAz�B�k�                                    By�_T  �          A
�\?��\?�
=AB��=Bj�?��\@�G�@��
BT�RB�#�                                    By�m�  
�          @�
=>\)���
@�B�C��f>\)@C�
@�\B��B��f                                    By�|�  �          A Q�?@  ?�=q@�=qB��3B�{?@  @�\)@�p�BU{B��f                                    By׋F  b          A  ?�Q�@�=q@��Bg{B���?�Q�@ָR@���B	\)B��3                                    Byי�  x          AG�?��@y��@�=qBj{B���?��@У�@���BQ�B��f                                    Byר�  �          A�
?��
@<(�@�
=B���Bj�?��
@�=q@�G�B)=qB��f                                    By׷8  �          A=q?��H@Dz�@�B�z�Bs�?��H@�  @��B'�HB�8R                                    By���  T          A�?�
=@Vff@�\)B}ffB}Q�?�
=@�ff@��Bp�B��{                                    By�Ԅ  
�          A
=?�
=@Q�@�p�B�(�B?�?�
=@���@���B@��B��f                                    By��*  �          Aff?�33?��HA�B�ǮB��?�33@�  @ٙ�BY(�B�8R                                    By���  �          A	p�?ٙ�@��AB���BN
=?ٙ�@��@�{BC��B��)                                    By� v  T          A\)?�{@
=A�
B�G�BB{?�{@�z�@�=qBD33B��H                                    By�  
�          A��?�{?���A�B�p�B133?�{@��\@��HBJffB��R                                    By��  F          A	?ٙ�@O\)@���B��{By(�?ٙ�@�
=@�B%Q�B�k�                                    By�,h  �          A�R?�{@"�\A�B�z�BW�?�{@�=q@θRB:Q�B��                                    By�;  T          A�H?z�@�@���B9��B��q?z�A33@Z�HA���B�ff                                    By�I�  �          A��>W
=@�{@��
B*z�B��R>W
=A�@=p�A��RB�u�                                    By�XZ  �          Az�>8Q�@�  @��HB1Q�B�>8Q�A�@N�RA�Q�B��q                                    By�g   �          A�R�#�
@��@�\)BA�RB��׽#�
A�@uA�
=B�u�                                    By�u�            A�
<��
@�z�@�33BD  B���<��
Aff@|��A���B��q                                    By؄L  �          AG���33@�p�A (�BZ�RB����33Az�@��A�33B��R                                    Byؒ�  �          A�H>#�
@�z�@��BVp�B��)>#�
A�@�  A�ffB��H                                    Byء�  T          A�\>�@\@��BO��B���>�A�@�  A�B�                                    Byذ>  �          A�?���@�\)@�BC
=B�L�?���A��@�G�A�G�B��=                                    Byؾ�  
�          A?��
@��
@�(�B2=qB��?��
A=q@Tz�A�Q�B��                                    By�͊  �          A�?��@Ϯ@�(�B=  B��?��Aff@l��A�p�B�p�                                    By��0  
�          A�\?�ff@�\)@љ�B+ffB��f?�ffA��@>{A�B��
                                    By���  
�          A�?O\)@�@��
B6ffB�.?O\)A\)@X��A��B��                                    By��|  "          A?p��@ᙚ@�\)B)G�B���?p��A{@7�A�
=B�=q                                    By�"  "          A=q?��
@ٙ�@׮B1��B���?��
A(�@Mp�A�{B�                                      By��  �          A��?�ff@�
=@��B8ffB��?�ffA��@`��A��B�
=                                    By�%n  �          A\)?��@�p�@�p�B5�HB�u�?��A�@\(�A��B�ff                                    By�4  	�          A  ?��H@�(�@���B#z�B�B�?��HA�R@1G�A��B��H                                    By�B�  T          A�@\(�@�\)@�p�B �B���@\(�A\)?�G�A�\B�                                    By�Q`  �          AQ�@O\)@�z�@���B	�\B��@O\)A��?�{A/�B��\                                    By�`  
Z          A�@dz�@�@��RA�=qB��R@dz�Ap�?�=q@�
=B�                                    By�n�  T          A33@J=q@���@EA�\)B�Ǯ@J=qA��\�Q�B��=                                    By�}R  �          A�@K�@�Q�@1�A��HB�k�@K�A\)�G���G�B���                                    Byً�  �          A�\@+�@�(�@�B��B�u�@+�A�?��A   B�W
                                    Byٚ�  T          A�R@N{@ڏ\@��B z�B�z�@N{A
=?��A�B�{                                    By٩D  "          A{@5�@�=q@��
B {B��)@5�A=q?��
A�B��{                                    Byٷ�  �          A(�?��
@�\)@�
=B#�\B���?��
A��@�HA��HB�k�                                    By�Ɛ  �          A{?�\)@�{@�\)B�B��q?�\)A=q@�A]�B��
                                    By��6  "          A�\@
=q@�ff@�ffB��B�=q@
=qA�@�AX��B�{                                    By���  "          Aff?ٙ�@�\)@�=qB�HB���?ٙ�A\)@�Ab�RB�B�                                    By��  T          Az�?�p�@���@�G�B\)B���?�p�A=q?У�A+�B�k�                                    By�(  �          A��@"�\@��
@�  B
=B��@"�\A��@�Ap(�B�33                                    By��  
�          A�R@�H@��H@�\)B�HB�@�HA��@
=qA`Q�B��=                                    By�t  "          A
=q@�@���@�p�B=qB�.@�@��@\)AqG�B�33                                    By�-  �          A	��@!�@��@��B*=qB���@!�@�\)@333A�B�Ǯ                                    By�;�  
�          AQ�?\@�z�@ ��A�33B��\?\@�\)�xQ����B��                                     By�Jf  �          A\)?��\@�
=?�{A4��B��)?��\@��Ϳ�
=�X��B��3                                    By�Y  �          A�?��
@��H@*=qA�p�B�B�?��
A   �Q���Q�B�ff                                    By�g�  �          A�@Q�@�p�@333A���B�@Q�@��Ϳ�R���B��f                                    By�vX  �          Az�@
=@�G�@r�\A�Q�B���@
=@�\)>�G�@@��B��                                    Byڄ�  �          Az�@�H@�\)@�p�A��HB�(�@�HA
ff?Q�@��B�                                    Byړ�  "          A{@��@�p�@�G�A�{B�W
@��A�?���@�=qB�#�                                    ByڢJ  �          A�R@=q@�@���A��RB�33@=qAz�?�@���B�.                                    Byڰ�  "          A�
@\)@��@���A���B���@\)Aff?�{@�  B�z�                                    Byڿ�  T          A�@
�H@�33@��HA��B��@
�HA�\?�ff@�33B�\)                                    By��<  T          A�?��RAG�@�z�A���B�#�?��RA�\>�(�@$z�B��                                    By���  b          A��@p�A�@��A�B�8R@p�A(�?0��@�G�B���                                    By��  x          A{@��A(�@��A�Q�B�(�@��A��>�@8Q�B��                                     By��.  �          AQ�@
=qA
=@�=qA�=qB�z�@
=qA  >�(�@&ffB��q                                    By��  	�          Aff?��HA�@mp�A��\B���?��HAff�L�Ϳ�p�B��
                                    By�z  �          Aff?�33A  @`��A���B�?�33A�H��(��(Q�B���                                    By�&   �          A�
@
=qA ��@���A�p�B���@
=qA\)?5@���B���                                    By�4�  �          A33?�\A   @�ffA���B��?�\A  ?\(�@�
=B�\)                                    By�Cl  �          A  @
=q@�{@��AB��=@
=qA33?k�@�=qB��\                                    By�R  T          A{@(�A ��@���A�33B��\@(�A��>�ff@1�B���                                    By�`�  T          A��?��RA ��@��A�  B�8R?��RA>��H@@��B�aH                                    By�o^  �          A�@(Q�@�G�@�33B�RB�
=@(Q�A  @�AMG�B��=                                    By�~  �          Aff@\)@��R@�(�A�z�B���@\)A��?�33@��B�33                                    Byی�  �          A�\?�z�A@�33AܸRB��=?�z�A
=?   @C33B��\                                    ByۛP  �          A�\?��A
�\@`��A��B��3?��AG����H�?\)B��R                                    By۩�  �          A�?�=qA�\@~�RA�
=B�Q�?�=qA��<�>W
=B��R                                    By۸�  �          A�?�A��@�  AˮB���?�A33=�G�?+�B��                                    By��B  �          A��?�z�A�@��HA��B�(�?�z�AQ�>8Q�?�{B�Ǯ                                    By���  �          AG�?���A�
@���A�p�B�u�?���A�>��?�\)B�ff                                    By��  �          A?�ffA�
@�Q�A˅B�\)?�ffA�\>\)?\(�B�Ǯ                                    By��4  �          A�@��A�@z=qA�G�B�aH@��A33<#�
=�\)B�{                                    By��  �          AG�@=qA=q@z�HA���B�  @=qAQ�    ���
B��H                                    By��  �          A�H@1�A=q@�A��
B�B�@1�A  ?\)@Tz�B��                                    By�&  T          Az�@�HA�\@�G�A���B��@�HA�R?aG�@��B�(�                                    By�-�  �          A�
@�@�p�@��
A�z�B�L�@�A��?��
@��HB�\                                    By�<r  �          A33?=p�@��@��Bp�B�8R?=p�A  ?�  A&ffB�\                                    By�K  T          A\)?
=@��@��B�B��?
=A�\@�\AZffB��
                                    By�Y�  �          A{>���@�G�@��RB(�B�  >���A��@p�ATQ�B���                                    By�hd  �          Ap�?�@�R@��B��B��?�A��@G�AZ�HB��                                     By�w
  �          A��>�G�@�  @�{B!B�
=>�G�A
=@"�\Aw33B��                                     By܅�  �          A�>��
@�Q�@��
B(��B���>��
A��@333A��B���                                    ByܔV  �          A\)���@�{@�p�B+  B�p����A(�@8Q�A��B��H                                    Byܢ�  �          A
=�8Q�@��@���B333B�ff�8Q�A��@Mp�A�G�B�\)                                    Byܱ�  �          A�
�5@�\)@�(�B:=qB�k��5Az�@`  A�\)B�.                                    By��H  �          AQ�8Q�@���@�(�B9Q�B����8Q�AG�@^�RA��B�aH                                    By���  �          AQ��(�@Ϯ@޸RB;�B��\��(�A�@c�
A��
B��\                                    By�ݔ  T          A�H��Q�@�p�@���B<=qB�����Q�A�
@b�\A���B�.                                    By��:  �          A녾���@��@��B>Q�B�  ����A
=q@fffA��B�aH                                    By���  �          Az὏\)@���@�
=BL��B�ff��\)A��@�33A�=qB���                                    By�	�  �          A�R���\@�p�@��B1�RB�ff���\AQ�@G
=A���B���                                    By�,  �          A����@�ff@�33BK(�B��ῇ�A�\@�G�A��B½q                                    By�&�  �          A
=?�\@Ӆ@�\)B5��B�33?�\AG�@S�
A�Q�B�ff                                    By�5x  �          A�\>��@ָR@�33B1�B�  >��A@J=qA���B��3                                    By�D  �          A{>aG�@�\)@���B0ffB�=q>aG�A��@E�A���B�#�                                    By�R�  �          A�?8Q�@�z�@��B:z�B�(�?8Q�A
�\@^�RA���B��                                     By�aj  �          A��?�@�z�@ٙ�B:��B�z�?�A
�\@^{A��\B���                                    By�p  �          A�?(�@�ff@أ�B9{B�33?(�A
=@[�A�B���                                    By�~�  �          AG�?#�
@�=q@ڏ\B<Q�B�k�?#�
A	@b�\A�Q�B�z�                                    Byݍ\  
�          A��=���@�@�
=B9
=B�33=���A
�\@X��A�33B���                                    Byݜ  �          Az�#�
@�\)@�Q�BN�HB�#׾#�
A  @�A�
=B�.                                    Byݪ�  �          Az�?W
=@��\@�z�BI�HB���?W
=A��@���A�z�B�u�                                    ByݹN  �          AQ�z�H@��@�B9Q�BŞ��z�HAz�@Z�HA��B�{                                    By���  �          A�ÿ�\)@��
@��
B?p�B��
��\)A
=@j�HA���B�33                                    By�֚  �          A�Ϳh��@�33@�z�B@��B��f�h��A�H@l(�A�Q�B�B�                                    By��@  �          A  �@  @ƸR@�{B;�B���@  A33@^{A�  B�.                                    By���  �          Az�.{@��@�\BG{B�8R�.{Aff@z�HA��B�L�                                    By��  �          Azᾔz�@\@߮BC��B�k���z�A33@s33A��B��f                                    By�2  �          A��:�H@�@��B>Q�B��\�:�HA\)@eA��B��                                    By��  �          A�R>8Q�@�G�@��
BB�RB���>8Q�A@mp�A��B��{                                    By�.~  0          A?B�\@��@���B;��B��f?B�\A{@]p�A��B���                                    By�=$  x          A�#�
@ƸR@���B;��B�#׼#�
A�R@\(�A��
B��                                    By�K�  �          A�H��z�@��
@ӅB8�B�=q��z�A��@UA���B��f                                    By�Zp  �          Aff�
=@���@�  B5=qB����
=A��@N�RA��B���                                    By�i  �          A��?�  @�ff@�=qB:��B�8R?�  Aff@^�RA���B�                                    By�w�  �          A��?�33@ȣ�@�z�B4Q�B�L�?�33A�@L(�A���B�k�                                    Byކb  �          A�H�W
=@���@��BGB�(��W
=A��@z�HA�Q�B���                                    Byޕ  �          A녿�\@��@�p�BO\)B�k���\Ap�@�A�{B�=q                                    Byޣ�  �          A�׾.{@�
=@߮BJB�aH�.{A@~{AЏ\B�aH                                    By޲T  �          A�
?�\@�ff@�  B9  B��?�\A��@U�A��B��                                    By���  �          A�þ���@��
@�Be33B�����@�33@�z�B�B���                                    By�Ϡ  �          A{��33@��
A ��Bw�HB�B���33@�33@�33B��B�u�                                    By��F  �          A(�?��@��@�z�B/\)B�Ǯ?��A�
@I��A�p�B�.                                    By���  T          A�?�p�@�ff@�\)B(  B���?�p�A
�H@8Q�A�=qB�ff                                    By���  �          A(�?���@�@��B�\B�Q�?���A��@(�A[33B��R                                    By�
8  �          A(�?ٙ�@�@���B ffB�aH?ٙ�A�
@ ��A�33B�L�                                    By��  �          A��?��R@�=q@�Q�B8�\B�\)?��RA�@Y��A��B�k�                                    By�'�  �          A(�?�z�@�33@�ffB7ffB�?�z�A�@UA�z�B��{                                    By�6*  �          A33?�z�@��\@�Q�B:�B��3?�z�A   @`��A�
=B�(�                                    By�D�  �          A�?�33@��H@�BQ��B��3?�33@���@�Q�A��B�=q                                    By�Sv  �          A�?��
@���@ڏ\BH��B�
=?��
@�(�@z�HA�
=B��=                                    By�b  �          A�?��H@�(�@�=qBF(�B�  ?��H@�ff@xQ�A��B�=q                                    By�p�  �          A=q?�{@\@�Q�B-p�B��?�{A Q�@<��A��B���                                    By�h  �          A��@*=q@�p�@��Bz�B���@*=qA(�?��HA��B�W
                                    Byߎ  �          A�@0  @��H@�p�BG�B�=q@0  A�\?�A=qB�Ǯ                                    Byߜ�  �          A�
@�R@�\)@�\)B(�B��=@�RA�H?�G�A:{B�L�                                    By߫Z  �          A  @�\@�(�@�G�B
=B��{@�\A��@�HA�(�B�z�                                    Byߺ   �          A�@G�@�ff@��RB#G�B��H@G�A   @(��A���B�aH                                    By�Ȧ  �          A
=@{@�z�@�ffBp�B���@{A ��@Ay��B�(�                                   By��L  �          A
ff@(�@Ϯ@���B\)B���@(�Ap�@Q�Ad��B�                                   By���  �          A
=q@z�@ȣ�@��RB�
B�ff@z�@��R@��A���B��                                     By���  �          A\)@!G�@�p�@�33B 33B�� @!G�@�@$z�A�p�B��                                    By�>  �          A
�R@
=@ҏ\@��
B�HB�z�@
=A��?�(�AQ�B�z�                                    By��  �          A�
@*�H@�p�@��B\)B�B�@*�H@��@"�\A�G�B��q                                    By� �  T          A�@C33@��@�=qB��B{��@C33@�  @(Q�A�  B�                                      By�/0  �          A  @%@\@�{B#  B��
@%@��
@,��A�ffB���                                    By�=�  �          A  @@��@�(�@�
=B�RBQ�@@��@��H@�RA���B��                                    By�L|  �          A�
@G
=@�p�@��\B�B|�H@G
=@��@Ay�B��=                                    By�["  T          A
�H@c�
@�ff@��RB�RBl�\@c�
@�=q@z�AxQ�B���                                    By�i�  T          A
�H@P��@�G�@�=qB{Bvz�@P��@�@��A33B�                                      By�xn  �          A(�@A�@���@�p�B�HB}=q@A�@��
@��Av�\B��{                                    By��  �          A
=@'�@��@���Bp�B�8R@'�@�33@!G�A�
=B���                                    By���  �          Az�?�@�{@�B-z�B���?�@�Q�@7
=A�{B�B�                                    By�`  �          A(�>�z�@�G�@�
=Bg{B�Ǯ>�z�@��H@��B
=B�\)                                    By�  �          A\)?�\)@�@�BRp�B���?�\)@�@�
=A癚B�.                                    By���  �          A�@c�
@���@�ffB p�Bb��@c�
@���@1�A��B~G�                                    By��R  �          A�
@r�\@�
=@�\)B��B]�@r�\@��
@#�
A��HBx                                      By���  �          A�@5�@��@��B(  B|�
@5�@�@8��A�  B�=q                                    By��  
�          A  @y��@��\@���B	��B`�H@y��@��@G�A\z�BwG�                                    By��D  �          A  @��H@�G�@�{B�
B[33@��H@�ff?�Q�AR�\Bq��                                    By�
�  �          A  @��
@���@�z�B�RBZ�@��
@�p�?�33AO\)Bpp�                                    By��  T          A�
@��@�Q�@��
A�  BE33@��@��?�G�A@  B\��                                    By�(6  �          A�
@���@�Q�@���A��HB>z�@���@�p�?���A{BS�                                    By�6�  �          A�@��@�p�@��RA�(�B=�\@��@��?�z�A4��BT�R                                    By�E�  �          A�@��@���@�=qA��B:�@��@ҏ\?���AF�HBSff                                    By�T(  T          A�H@�=q@�@~�RA�33B;�H@�=q@�=q?���A=qBQp�                                    By�b�  �          A�@U@�33@��B�
Bh{@U@���@
=Ay��B                                      By�qt  �          A=q?@  @�{@�  B>{B�k�?@  @���@S33A��B��                                     By�  �          A=q?0��@�p�@�\)BH=qB��R?0��@�@g�A��
B�\                                    By��  �          A��>���@�p�@�G�BD��B���>���@�@e�A�=qB�\)                                    By�f  �          A����33@��
@��B:=qB����33@�\@R�\A��B�B�                                    By�  �          AQ�
=q@�ff@�{BB  B�k��
=q@�
=@^�RAǅB�W
                                    ByẲ  �          A\)>��@��@��BB�HB���>��@�p�@^�RA���B��                                    By��X  �          A  >��
@��@���B;�
B�p�>��
@�@QG�A��B��                                    By���  �          A�?��@��
@�{BC�\B�=q?��@���@a�A�\)B��R                                    By��  �          A�H>��@��R@ə�BI=qB���>��@�G�@l(�A֏\B���                                    By��J  �          A�\>��@�
=@θRBQ33B�(�>��@�z�@|(�A��B�p�                                    By��  �          A?�  @�\)@�=qBD��B��?�  @�=q@mp�A�
=B�Ǯ                                    By��  T          A�@�H@�z�@�B4z�B���@�H@陚@R�\A�G�B�\)                                    By�!<  �          AG�?��H@���@�33B<�\B��H?��H@�(�@\��A�B��                                    By�/�  �          A�
?�{@��\@�  B1p�B�.?�{@���@C�
A��\B��)                                    By�>�  �          A�@%@��
@���B$G�B�z�@%@��@.{A��\B��f                                    By�M.  �          A�@0  @���@���B =qB��@0  @���@'
=A�(�B��=                                    By�[�  �          A33@.�R@��@�33B  B���@.�R@�@Q�A��B�B�                                    By�jz  �          A{@$z�@��R@��B'ffB��{@$z�@��@333A�B�z�                                    By�y   T          A Q�@QG�@�\)@��B33Bl�H@QG�@޸R@33A���B��R                                    By��  �          @��@?\)@�@���B\)B|�H@?\)@�z�?˅A:{B��                                    By�l  �          @��
@XQ�@���@�G�Bp�Bc  @XQ�@�(�@�A��B{�                                    By�  �          @��\@Fff@�=q@�G�B  Bs�@Fff@��?�p�AjffB���                                    By⳸  �          @�@W
=@�\)@��RB�Be33@W
=@��@��A�
=B}                                      By��^  �          @���@K�@�
=@�Q�BBj�@K�@ҏ\@A�  B�k�                                    By��  �          @��H@Z=q@��@�ffB=qBZ��@Z=q@���@�A�ffBu��                                    By�ߪ  �          @�\@�p�@�ff@vffA�(�BG��@�p�@��?���AD  B^
=                                    By��P  �          @�R@0��@�@�z�B<�BbQ�@0��@��R@Tz�A֏\B�W
                                    By���  �          @��?�33@���@�  B.Q�B��\?�33@�
=@+�A��B�8R                                    By��  �          @�Q�@��@���@���B33B���@��@�33?��RAx  B�B�                                    By�B  �          @�  @L��@�\)@�Q�B#\)BY�@L��@�\)@(Q�A�p�BwQ�                                    By�(�  T          @�p�@mp�@^�R@��HB9��B,��@mp�@�G�@c�
A�RB[��                                    By�7�  
�          @��@��R@p  @�
=BQ�BQ�@��R@�ff@)��A��B?ff                                    By�F4  �          @��H@�{@��R@3�
A��
B  @�{@�
=?s33@�\B)                                    By�T�  �          @��@���@�33?���A%��B33@���@��׿����ffB{                                    By�c�  �          @�
=@�{@���?�\AU�B-�R@�{@���p��0  B3�
                                    By�r&  �          @�Q�@��@��?�33A'33B$  @��@�z�&ff��G�B'p�                                    By��  �          @�
=@�\)@�p�?�AH��B8(�@�\)@��Ϳ\)���B<�
                                    By�r  �          @���@�z�@���?�{A^{B<z�@�z�@�=q��
=�HQ�BBQ�                                    By�  �          @���@�  @��
@�Aq�B6��@�  @���W
=���
B>33                                    By㬾  �          @��R@���@�p�@	��A��B<��@���@��\��G��Y��BD�H                                    By�d  �          @�  @��@���?�z�AEBFp�@��@�
=�333����BJ(�                                    By��
  �          @��R@�{@�  ?���A(�BEQ�@�{@�������BE(�                                    By�ذ  �          @�@�p�@ȣ׿h������Ba�\@�p�@�{�P����\)BR{                                    By��V  
�          @�\@���@���?�{A�B\p�@���@�Q쿙���ffB\�                                    By���  T          @�(�@���@���@	��A���B9=q@���@�
==L��>�(�BB�
                                    By��  �          @��
@�
=@�\)@%A�
=BE�@�
=@�=q>\@@  BQ��                                    By�H  �          @�z�@�\)@�p�?�z�AP��BC�@�\)@�z�
=q��BH�                                    By�!�  �          @�33@�Q�@��H?���AI�BA33@�Q�@��������RBEp�                                    By�0�  �          @��@��H@�=q?��
AffB3�@��H@��ͿY�����
B5ff                                    By�?:  �          @�p�@���@�(�@A}�B0�
@���@�G����B�\B9��                                    By�M�  �          @�\)@��@���@.�RA��HB!�@��@�\)?L��@�33B2G�                                    By�\�  �          @�\)@���@�?�(�AuBA��@���@�\)�B�\���HBH�H                                    By�k,  �          @���@�=q@�G�?�G�Af�HBJ�@�=q@�녾Ǯ�H��BPz�                                    By�y�  �          @У�@Z�H@p  @��BffB>�\@Z�H@�Q�@�A�B^�                                    By�x  �          @��@��\@e�?�z�A4z�B  @��\@p  ��\)�0  B=q                                    By�  �          @�ff@�Q�@<(�?E�@陚A�@�Q�@@�׾����qG�A�z�                                    By��  T          @Ǯ@�(�@=p�@�A�ffA�Q�@�(�@j=q?���A'\)B��                                    By�j  �          @�
=@�z�@vff@333A��HB$�H@�z�@��
?��A!�B9��                                    By��  �          @�{@��@�33@HQ�A�B2ff@��@�\)?���A<��BH�\                                    By�Ѷ  �          @��@�ff@��\@Z=qA�B0��@�ff@��?�\)A]�BIff                                    By��\  �          @�ff@`  @�ff@��B33BH�@`  @��@ffA��RBf                                      By��  �          @���@:=q@��R@�=qB�RBiQ�@:=q@��@�RA�{B�33                                    By���  �          @���@aG�@��@��B33BMff@aG�@�(�@	��A�Bg                                    By�N  �          @���@5�@��R@��
B��Bk�
@5�@�  @�\A�{B�u�                                    By��  �          @�ff@z�@�\)@��HBp�B��@z�@ٙ�@p�A�(�B�33                                    By�)�  �          @�\@���@@���B%33A�ff@���@q�@G�A�z�B%33                                    By�8@  �          @�ff@���?^�R@��B{AG�@���@��@qG�BA�G�                                    By�F�  �          @���@��#�
@�33B1�
C�  @�?Ǯ@��\B%  A��                                    By�U�            @��H@�  �%�@�ffB �RC�aH@�  �Tz�@�33BA{C�xR                                    By�d2  �          @�ff@=p�@3�
@�ffBI�HB-�H@=p�@�ff@c33BffBa=q                                    By�r�  �          @��@��@l(�@��B?=qBl@��@�@A�A�ffB�                                    By�~  �          @�  @	��@w�@���BC  Bt  @	��@�
=@P��A�  B�#�                                    By�$  T          @��
>W
=@L(�@�{Bz��B�k�>W
=@�Q�@�=qB#�B�ff                                    By��  �          @�Q�5@�@�=qB�aHB�\)�5@��@�{BE�HB���                                    By�p  �          @�׿h��?�\)@���B�W
B��Ϳh��@�{@�p�B\B̔{                                    By�  �          @��Ϳ.{?�p�@�RB�W
B�p��.{@�(�@�z�Bc=qB�Ǯ                                    By�ʼ  �          @����?�(�@�  B��B��\���@��
@�{BdQ�Bę�                                    By��b  �          @�{?\(�@�\@�G�B��qB��f?\(�@��R@��RBH
=B�aH                                    By��  
�          @�z�?��@33@��B��HB}
=?��@��\@�Q�BP�B��                                    By���  
�          @�?}p�@*=q@�G�B��B�B�?}p�@���@��\B=33B��H                                    By�T  T          @��?�ff@-p�@�p�B�  B���?�ff@��@�ffB9  B�{                                    By��  �          @�G�@��@@  @���BrG�BZ  @��@�@��B&�RB�{                                    By�"�  �          @�ff@)��@<��@ȣ�Bf�B@G�@)��@�G�@���BBx�H                                    By�1F  �          @�?�@>�R@љ�BwffBi{?�@��@�Q�B)�B��                                    By�?�  �          @�?�@I��@�ffBz��B���?�@��
@�33B*�B�z�                                    By�N�  �          @�G����@L(�@�ffB}G�B��)���@���@��HB*��B��H                                    By�]8  �          @�
=���@5@ۅB�#�B�
=���@�(�@��
B8ffB�p�                                    By�k�  �          @��
��ff@8Q�@أ�B�W
BŽq��ff@�(�@���B6�B�                                    By�z�  �          @�=q���\@4z�@��
B��fB�Q쿢�\@���@���B3z�B�L�                                    By�*  �          @��H����@��@���B�#�B�{����@�p�@��RB@�\B֊=                                    By��  �          @��H���R@33@��B�(�B�{���R@��H@���BCp�B�
=                                    By�v  
�          @�  ���@(�@�B�B�zῧ�@�@��B?\)B�G�                                    By�  T          @�p���  @*=q@ҏ\B�=qB���  @��H@�{B9p�B�=q                                    By���  �          @����R@�R@�(�B�  B��q���R@�ff@���BC�HB��                                    By��h  �          @��Ϳ��@33@��HB�� B��{���@�  @��\BA�BٸR                                    By��  �          @�33���@{@�=qB���B��)���@���@�  B?33B�Q�                                    By��  �          @���ٙ�@{@ҏ\B��B��q�ٙ�@�p�@�33BB{B�                                    By��Z  �          @��H��?��@�G�B���C���@w
=@���BMG�B��
                                    By�   �          @����9��?�=q@�z�B~  CaH�9��@`��@�Q�BI�C�=                                    By��  �          @���XQ�?���@ƸRBp�C!��XQ�@N�R@��BD��CQ�                                    By�*L  �          @�
=�A�?�{@���BzG�C���A�@a�@�Q�BG(�C�f                                    By�8�  �          @�ff�7
=?��R@�B}ffC���7
=@j=q@��BF��C �                                    By�G�  �          @��Dz�?���@��HBxC�\�Dz�@^{@�
=BF�C�\                                    By�V>  �          @����G
=?�Q�@�33B|�C
�G
=@\(�@�Q�BLC(�                                    By�d�  �          @��ÿ��
@�@�(�B���C�ÿ��
@�=q@��RBE�
B���                                    By�s�  �          @��H�xQ�@0  @θRB�z�B��)�xQ�@��H@��HB7�Bʣ�                                    By�0  �          @��
���\@2�\@�ffB��{B�  ���\@��
@�=qB6
=Bˊ=                                    By��  �          @�p���=q@3�
@�Q�B�k�B�녿�=q@���@��
B6(�B̽q                                    By�|  �          @�\)��{@<(�@θRB}ffB噚��{@�Q�@���B0G�B�\)                                    By�"  �          @�Q쿬��@5�@�G�B��qB��ÿ���@�@�z�B4�BҞ�                                    By��  �          @�G����@+�@���B���B�=���@�=q@��B:��B�                                      By��n  �          @�(��u@��@ָRB�p�B���u@��H@���BMG�B��H                                    By��  �          @�\)�!G�?�  @أ�B���B�ff�!G�@p  @��\Bc�B���                                    By��  �          @޸R��\)?�  @�33B�
=B�aH��\)@z�H@��HBWp�B��                                    By��`  �          @ۅ��=q?ٙ�@�ffB�aHCٚ��=q@u�@�
=BSz�B��f                                    By�  �          @��Ϳ�=q?�=q@�z�B��HC녿�=q@z�H@��
BLz�B�                                      By��  �          @޸R��
@�@�=qB��fCh���
@��\@��BC�RB�=                                    By�#R  �          @�{��(�?��@��
B��CaH��(�@|��@��HBI�\B�\                                    By�1�  �          @�(�� ��@�@�ffB���C�� ��@�@��\B>ffB�                                     By�@�  �          @�Q��33@�@���B��C
���33@���@��RB@�
B�33                                    By�OD  �          @�  �{?�(�@ǮB|ffCc��{@~�R@�{B?��B��                                    By�]�  �          @޸R�˅@   @�ffB���C }q�˅@��\@�z�BK��Bޙ�                                    By�l�  �          @�Q쿞�R?��R@��HB���B��)���R@��
@���BPp�BՔ{                                    By�{6  �          @��H�O\)?�
=@أ�B���B��O\)@��@��RBV�HB�aH                                    By��  �          @�\)��  @��@��HB���B��;�  @��@�BL�RB���                                    By蘂  �          @�녽�G�@
=@�\)B��B���G�@�Q�@��
BS=qB��                                    By�(  �          @���?5@   @�  B�aHB�\)?5@���@���BC�\B�G�                                    By��  �          @�\>�G�@&ff@�B�aHB���>�G�@��
@�B@�B�33                                    By��t  �          @�(��,(�?���@�B��C�q�,(�@aG�@�=qBM�
B�Ǯ                                    By��  T          @�ff��ff@�
@��B�L�C=q��ff@�p�@��\BK\)B�Ǯ                                    By���  
�          @�  ��Q�@ff@�
=B�L�C ���Q�@�\)@�(�BL=qB߮                                    By��f  �          @��ÿ��R@
�H@أ�B��B�(����R@���@��BL��B�G�                                    By��  �          @�׿�z�@p�@�ffB��B�3��z�@���@�  BDG�B�p�                                    By��  �          @�R��ff@��@ָRB�B����ff@�  @��
BL��B�#�                                    By�X  �          @��\@Q�@�  B��B��f�\@�  @��BM��B�B�                                    By�*�  �          @�G���
=@{@�G�B��=B�����
=@��H@�BL�\B�z�                                    By�9�  �          @陚���@�R@�p�B��RCaH���@��@��BG{B�p�                                    By�HJ  �          @��H�Ǯ@#33@�p�B�B��f�Ǯ@�33@�ffBA(�B�aH                                    By�V�  �          @�R�   @@  @���B��3B�\�   @�=q@�p�B;�B�\)                                    By�e�  �          @�(���z�@�@޸RB���C&f��z�@�33@��BK�B�W
                                    By�t<  �          @���<(�?��R@�G�B��)C5��<(�@mp�@�p�BQ�HC c�                                    By��  T          @��H�(��?ٙ�@�B��
C+��(��@}p�@�\)BR��B�L�                                    By鑈  �          @�33�'
=?�\)@�
=B��fC8R�'
=@x��@ə�BUp�B�Ǯ                                    By�.  �          @�����
@p�@�B�aHC
=��
@�@�=qBL�RB��                                    By��  �          @�=q��@   @�RB��qB�ff��@��R@�Q�BH��B�                                      By�z  �          @��\���@G�@�Q�B��{C \)���@�  @�(�BNB�u�                                    By��   �          @�녿���@�H@�G�B�8RB�Q쿰��@���@��
BN\)B�                                      By���  �          @�{�p��@�
@�  B���B�=q�p��@���@��
BSz�B�p�                                    By��l  �          @�  �!G�@+�@�\)B�Q�BΨ��!G�@��
@�\)BJz�B��)                                    By��  �          @��H�L��@5@��B��\B��L��@�G�@�  BH
=B�#�                                    By��  �          @�?z�H@;�@�z�B��B��?z�H@���@�33B;��B�u�                                    By�^  �          @��H�У�@O\)@�\)Bo=qB�G��У�@�=q@��B)�RBה{                                    By�$  �          @�p��e@<��@��BX�C���e@��H@�(�B!z�B�.                                    By�2�  �          @�\���R@<��@���B<�C�R���R@�=q@�=qB(�C��                                    By�AP  �          @���n�R@#33@ȣ�BX��C�3�n�R@��@�(�B&ffC@                                     By�O�  �          A Q��_\)?�p�@�{Bo��Cn�_\)@��\@��RB?�
C��                                    By�^�  �          @�p��Z�H@��@أ�Bk33C�)�Z�H@�G�@��RB8�C ��                                    By�mB  �          @���8Q�@*=q@��Bpp�C	J=�8Q�@�p�@��
B6��B�=q                                    By�{�  �          @�z��0  @��@߮By�RC
�3�0  @�\)@�(�B@�RB��                                    Byꊎ  �          @���Dz�@�\@�p�Bt=qC=q�Dz�@�33@�33B>�RB�Q�                                    By�4  T          @��R�E@{@�  Bu�
C:��E@��@�{BA{B�=q                                    By��  T          A (��S33@�@�
=BrffCG��S33@�ff@�ffB@\)C 0�                                    By궀  �          A z��P  ?�p�@�=qBv�HC���P  @��H@��HBE��C s3                                    By��&  �          AG��>�R?�33@�  B�
=CxR�>�R@�=q@ə�BM=qB�p�                                    By���  �          A��e?�(�@�Q�Bq��Cp��e@s�
@�(�BE�HCG�                                    By��r  �          A ���Y��?�(�@�Q�Br�HC޸�Y��@���@���BC\)C�q                                    By��  �          A ���j�H@z�@ۅBiC���j�H@��H@�(�B;C�                                    By���  �          @�
=�vff@33@ҏ\B^��C+��vff@�
=@���B0�C^�                                    By�d  �          A��qG�@�@��HBgC���qG�@�G�@�(�B;
=C                                      By�
  �          @�\)�qG�?��
@ٙ�Bj{C�q�qG�@q�@�p�B?�C޸                                    By�+�  �          A ���~{?�p�@�\)Bbp�C� �~{@|��@���B8  C#�                                    By�:V  �          A
=��G�@$z�@ϮBQ�\C{��G�@�@�z�B$��C\                                    By�H�  �          A  ��Q�@{@�
=B[�Ch���Q�@���@���B-�HCW
                                    By�W�  �          A�����?�p�@�(�BX��C�����@z=q@��RB1�C
#�                                    By�fH  �          AG����\?�=q@ʏ\BF33C .���\@j�H@�\)B$��C&f                                    By�t�  �          A{��p�?�33@�BS�
C����p�@u�@�G�B/�C�H                                    By냔  �          Ap��7
=@Mp�@ۅBgC�q�7
=@�(�@�=qB-�HB�B�                                    By�:  �          A ���Z�H@:�H@��B`  Cs3�Z�H@���@�
=B+�\B��f                                    By��  �          A�>��R@���@�
=B`  B�\>��R@ʏ\@�G�B  B�Q�                                    By믆  �          A33=�Q�@�Q�@ϮBQp�B��f=�Q�@�
=@�{B�B�p�                                    By�,  �          A���Q�@�(�@�z�BR�B�
=��Q�@ҏ\@�z�BB��                                    By���  �          A�����
@�33@���B^�B��
���
@�z�@�G�B{B˙�                                    By��x  �          A
=�z�@|��@�
=B^��B��)�z�@���@��B ffB�Ǯ                                    By��  �          A��X��@Z=q@ϮBU�RC޸�X��@�{@�Bz�B�k�                                    By���  �          A z��XQ�@\(�@��HBR�HC���XQ�@��@���B�RB�z�                                    By�j  �          A Q��^{@E@ʏ\BV�C
Y��^{@�=q@�(�B#G�B�z�                                    By�  �          A���W�@fff@�p�BQ��C
�W�@��H@�=qB33B�                                     By�$�  �          A\)����@X��@�Q�BG��C޸����@��\@��BQ�C T{                                    By�3\  �          A Q��hQ�@XQ�@�BM  C	��hQ�@�G�@��BQ�B�z�                                    By�B  �          @�z����@W
=@�(�Bdz�B�����@��\@��
B)G�B�k�                                    By�P�  �          A����@�@ҏ\BY=qB����@�p�@�=qB�B�Ǯ                                    By�_N  �          A�ý�G�@�33@�33B8z�B�{��G�@�{@�G�A�\)B���                                    By�m�  �          A��=p�@�z�@�BOz�BÔ{�=p�@�@��BG�B��q                                    By�|�  �          Az����@�33@��BV�B�.���@�ff@�p�B�
B��H                                    By�@  �          A	G��s�
@-p�@���BcQ�C�
�s�
@��@��B4CB�                                    By��  �          A���^{@`  @�\)B[{C���^{@��
@�p�B&Q�B��q                                    By쨌  T          A��W
=@s�
@�z�BR\)Cp��W
=@��@�Q�B
=B�Q�                                    By�2  �          A���E@�@���BO��B��f�E@�
=@��HB�B��H                                    By���  �          AG��q�@�G�@�
=BJffCh��q�@Å@�{B��B�{                                    By��~  �          A(��.{@���@˅BIQ�B�(��.{@���@���B33B�Ǯ                                    By��$  �          A z����@�  @�=qB:�
B䞸���@�ff@��A�ffBڙ�                                    By���  �          AQ��Mp�@~{@���BS��C ��Mp�@��@��
B�B�z�                                    By� p  �          A  ���H@/\)@���B]��C33���H@���@\B1��CL�                                    By�  �          A  ��p�?�ff@�BV��C%���p�@R�\@ҏ\B<�C��                                    By��  �          A����
>�(�@޸RBL
=C/�����
@Q�@��B<G�C�                                    By�,b  �          A33���ýu@�=qBF�RC4�{����?�{@��B<�RC"(�                                    By�;  �          A����33���@�Q�B6ffC9+���33?�p�@��B2�
C(��                                    By�I�  �          @�����Q�>u@�=qBE
=C15���Q�?�{@���B7�RC�
                                    By�XT  �          @���z�5@��BS�C=33��z�?s33@�=qBQ\)C'�                                    By�f�  �          @˅��Q쿆ff@��BC33CB����Q�>��R@��HBI{C/��                                    By�u�  �          @��H��(���ff@�\)BO\)CK����(��\)@�  B]�C5��                                    By�F  �          @�z��������@���BPffCG޸���>\)@ǮBZC2:�                                    By��  �          A	��\)��
=@ۅBS�CD���\)?�@�\)BYp�C.
=                                    By���  �          AG���(��AG�@���B6  CQO\��(���
=@��
BM{C@Q�                                    By��8  �          @�  ��Q���
=?���A���C}\��Q����@3�
B�Cz��                                    By���  �          @���XQ����@eA�Q�Ck�{�XQ��s�
@�=qB-�\Cdff                                    By�̈́  �          @ȣ��Q��5@9��BQ�Cf��Q��Q�@dz�BEG�C[E                                    By��*  w          @Vff>.{@Mp�<�?z�B��)>.{@Fff�W
=�s33B���                                    By���  �          @p��
=?�?Q�B4
=C)�
=?E�?#�
B��B�                                    By��v  �          @��=�G�@�?��\A�33B���=�G�@��
=�Q�?�Q�B��                                    By�  �          @ʏ\?�33@�z�#�
���
B��?�33@�����w�B��H                                    By��  �          @��H>�33@��?&ff@�G�B�  >�33@�(��fff���B���                                    By�%h  �          A
=?�{AQ�=u>�
=B���?�{@�\)�
=q�m�B�8R                                    By�4  �          @�\)@�
@��;����8Q�B��f@�
@�������HB�aH                                    By�B�  �          @��H��(�@�p�@fffA�ffB�녿�(�@��?��Aw�B��H                                    By�QZ  �          @�33�Q�@z=q@���B?��C��Q�@���@���B\)B��                                    By�`   �          A
=��p�@AG�@�p�BL��C
��p�@�  @��
B%z�C��                                    By�n�  �          A�R��녿
=@�BH33C:
=���?���@�33BD�
C'��                                    By�}L  �          A
=��=q?�@�G�BQ\)C!8R��=q@j�H@љ�B5z�Ck�                                    By��  T          A�
����@��@�\)BR{C�H����@�33@�z�B4=qCxR                                    By  �          AQ����R?ٙ�@�\)BQ��C#p����R@j�H@��B7�HCE                                    By�>  �          A ������?�
=@�z�BOQ�C&�����@\(�@�B8�RCG�                                    By��  �          A z���33?8Q�@��BOQ�C-G���33@.�R@�33B>�C޸                                    By�Ɗ  �          A$Q��׮�h��@�ffBB�C;���׮?�\)@�p�BA33C*�\                                    By��0  �          A&�\��z῕@�\)B7�C=E��z�?Q�@���B9\)C-xR                                    By���  �          A(z���\)���
@�\B0�\C;ٚ��\)?fff@�33B1{C-#�                                    By��|  �          A(Q������u@�G�B(33C5������?˅@ۅB#
=C(}q                                    By�"  �          A$���ٙ����@陚B5
=CG�{�ٙ����H@�BA��C8#�                                    By��  �          A"=q��\)�C33AQ�B[
=CS� ��\)�Tz�Az�Bp\)C=p�                                    By�n  �          A%���\)����@�B6z�C_�q��\)�>�RA�BZp�CQ�                                    By�-  �          A&�H���
�H@���B.p�CD�
����\)@�
=B8\)C6:�                                    By�;�  �          A*ff�������@��B.�C[�{����4z�AffBN\)CM�                                    By�J`  �          A+
=���
�*�H@�33B0�CH�{���
�8Q�@�G�B=�RC9�=                                    By�Y  �          A.�H��\��Q�@�  Bz�CS����\�*=q@�p�B233CG��                                    By�g�  �          A2{����@�ffB33CY.����  @�{B �CP)                                    By�vR  �          A1���{���@�
=BffCR(���{�9��@�B!Q�CH{                                    By��  �          A2=q�
�\�s33@�=qA���CK�3�
�\�{@��HBz�CBff                                    By  �          A1����\)@��HA���CB@ �����\@�=qA���C;\)                                    By�D  �          A2ff��R���@���A�
=C:k���R>��@��A���C1}q                                    By��  �          A5���p���p�@���BC6Y��p�?��@�G�B�
C+��                                    By￐  �          A5����?0��@��HB�
C/�=��@�H@�ffB�HC$�f                                    By��6  �          A5���z�?=p�@�(�B
=C/5��z�@{@�\)BC$B�                                    By���  �          A6�R�  ?Q�@�\B&Q�C.� �  @*=q@�z�B�C"��                                    By��  �          A6�H��?��
@��
B.
=C,�)��@;�@��
B ��C ff                                    By��(  �          A9��
=?Y��AB2G�C.��
=@333@��B&  C!�                                    By��  �          A9p���ff=�A�RB:�\C3���ff@
�HA{B2z�C$��                                    By�t  �          A9G��p��G�A33B5  C9���p�?�p�A=qB3p�C+T{                                    By�&  �          A9�����RA   B.=qC<z���?=p�A ��B/�C.�3                                    By�4�  �          A:�\�����  @�Bz�C=�)���>�{@�=qB"(�C1�=                                    By�Cf  �          A<Q�����{@��\B'��C68R��?Ǯ@�B#C)޸                                    By�R  �          A<  � ��@ ��A�\B0G�C"�3� ��@�z�@�=qBffC^�                                    By�`�  �          A;
=��\)?�  A33B9=qC+\��\)@O\)@��B*=qC�H                                    By�oX  �          A;���{?}p�A	G�B;�C,����{@@��Ap�B.z�CE                                    By�}�  �          A<���   ?�Q�A	�B:�\C+���   @L(�A ��B,
=C@                                     By���  �          A<��� z�?��A	G�B:��C,�
� z�@C33Ap�B-�C33                                    By�J  �          A<Q����
?���A
=B>33C+�
���
@I��A�RB/�HC.                                    By��  �          A<(���G�?�{A�
B?��C+޸��G�@H��A�B1��C
=                                    By�  �          A<Q���\)?���A��BA�
C,#���\)@G
=A��B3�RC)                                    By��<  �          A<(���?���A	��B<33C,J=��@C�
AB.��C�                                    By���  �          A<��� ��?\(�A	G�B:�RC-�� ��@5AffB.��C �
                                    By��  �          A=���G�?uA	B:�C-33�G�@<(�AffB-�C                                       By��.  �          A=���\?J=qAQ�B8ffC.� ��\@/\)A��B-{C!p�                                    By��  �          A:�H�\)?��A��B4�C/���\)@   @�p�B*��C#�                                    By�z  �          A:=q���>�ffAffB1�C0����@�
@��\B)33C$h�                                    By�   �          A:ff�
{?�\@���B(G�C0���
{@�\@�\)B��C%(�                                    By�-�  �          A:ff�
=?Q�@�
=B&
=C.�
�
=@$z�@�\B�HC#�                                    By�<l  �          A:�R�	�?Tz�@�=qB(Q�C.z��	�@&ff@�p�B
=C#=q                                    By�K  �          A;��(�?fffAQ�B3�\C-��(�@0��@�33B((�C!�                                    By�Y�  �          A;�
� Q�?�Q�A�
B933C+�{� Q�@E�A   B+�RC�q                                    By�h^  �          A:�R���?��RA
{B?
=C*�f���@J=qA{B0��C��                                    By�w  �          A:ff��{?\Ap�BEQ�C(p���{@^{AQ�B4��C�R                                    By�  �          A:ff��?��A��BD�RC*B���@N�RAz�B5C�                                    By�P  �          A;�
���?�A
�\B>{C08R���@��A�B4�\C"�\                                    By��  �          A;\)���?�z�AG�BC��C'�����@eA�B2�C}q                                    By�  �          A:�\��@�
A=qBG  C$&f��@~�RA\)B3G�C0�                                    By��B  �          A=�����
@�\A��BG�C$����
@�  AB3C�=                                    By���  �          A;�
��@!G�A�B@(�C!E��@�z�@��RB*�Cs3                                    By�ݎ  �          A;�
��  @
=A�B?�HC"����  @�
=@��B+z�C�H                                    By��4  T          A<(�����@'
=A��BIffC�)����@�G�A  B2p�C+�                                    By���  �          A;���{@4z�AQ�BI{C����{@�
=A�HB1  C                                    By�	�  �          A;�
��  @L(�A��BD  CxR��  @���@�{B*z�C5�                                    By�&  
�          A;33��R@�\A�BA�HC$����R@x��AG�B/ffC}q                                    By�&�  �          A;�
��\)?�{A(�BA(�C+�\��\)@>�RA��B4\)C�H                                    By�5r  �          A<����Q�@��AG�BB  C$���Q�@�  A�\B/(�C�q                                    By�D  �          A<  ���
@{A{BD=qC#=q���
@��\A
=B0C�                                    By�R�  �          A;33��=q@
=qA�BE�C#���=q@���A33B1�C=q                                    By�ad  �          A:�R��z�?�Ap�BE  C&��z�@l(�A�
B3C}q                                    By�p
  �          A:�\��?У�Az�BCz�C'�R��@]p�A�B3�C0�                                    By�~�  �          A7����?�z�A�BF
=C)����@N�RA�B7G�C0�                                    By�V  �          A8z���ff?��RA{BI��C(W
��ff@U�A�B:(�C(�                                    By��  �          A8����(�?�z�AffBJ(�C&�H��(�@`  A��B9��Cٚ                                    By�  T          A:�R��@(�A33BG�
C!{��@�Q�A�B333C\                                    By�H  �          A>=q��\@(�A�BK��C ����\@�=qA(�B7  C�
                                    By���  �          A?
=��=q@�HA��BM  C!{��=q@��A	��B8Q�C��                                    By�֔  �          A?33��\)@1�Az�BK��CB���\)@��A  B5p�CO\                                    By��:  �          A@  ��G�@Tz�A��BKC�f��G�@�A�RB2ffC�H                                    By���  �          AB=q��  @O\)A=qBJp�C(���  @��AQ�B2
=C�)                                    By��  �          A@  �ᙚ@J�HA�RBG��C�\�ᙚ@��AG�B/�
C��                                    By�,  �          A>{��ff@\)A33BC��C:���ff@��@�{B'��Cff                                    By��  �          A8�����@��
AQ�BM��C����@��
@�(�B,��CǮ                                    By�.x  �          A4Q���  @���A  B<{CW
��  @�(�@�33B
=C=q                                    By�=  �          A   ��33@�
=@���B��Ch���33@���@���AٮC��                                    By�K�  �          A&�\���H@��R@�(�B4z�C�����H@��@�=qB�\C�f                                    By�Zj  �          A$  ��
=@���@��B<�C�3��
=@�\)@љ�BQ�C33                                    By�i  �          A  ���@���@�=qA��C�����@�G�@k�A��B��                                    By�w�  �          A�
��Q�@�G�@�G�A�ffC����Q�@��
@[�A��B��)                                    By�\  �          A�\���
@�z�@��HA��HC���
@�\)@^{A�ffB�.                                    By�  �          A{����@��@�  A�33C�3����@�\@I��A�
=B�z�                                    By�  �          A{���@�@���A홚C�
���@�  @Y��A��B�                                      By�N  �          A=q����@��H@��RA�{C)����@�z�@W�A��B�z�                                    By���  �          A�\��z�@��@�z�B ��C0���z�@ᙚ@u�A���B�Ǯ                                    By�Ϛ  �          A���@�ff@�  Bz�C�H���@Ӆ@���A�  C ff                                    By��@  �          A�H��G�@�ff@��B=qC����G�@�\)@�A�(�CT{                                    By���  �          Az����@�ff@��
B  C5����@�ff@��A�C8R                                    By���  �          A����@���@��RBG�C
�=���@���@�p�A��C                                    By�
2  �          A���z�@���@�33B�C33��z�@�ff@��\A��Ck�                                    By��  �          A�����@�{@��
B$��C�)���@�G�@���Bp�CQ�                                    By�'~  �          A����  @���@���B%��C!H��  @�(�@�
=B�C�H                                    By�6$  �          A������@�G�@�ffB&{C(�����@���@�Q�B�C��                                    By�D�  �          A �����
@�
=@�ffB/p�CJ=���
@�
=@�Q�BffC	�                                    By�Sp  �          A)���\)@��H@�Q�B4�HC�3��\)@�ff@��HB33C
�
                                    By�b  �          A(Q���  @�\)@�p�B,  C\)��  @�Q�@�{BG�C	T{                                    By�p�  �          A#���
=@�=q@�B*�RC����
=@���@�ffB(�C�
                                    By�b  �          A(������@��\@�B)��C�)����@��H@��
B{C	\                                    By�  �          A+�
����@��@�z�B.=qCs3����@�
=@�Bp�C
:�                                    By���  �          A,z����@l��@�33B3��C�����@���@أ�B  C�\                                    By��T  �          A-���p�@^{@�p�B=\)C�f��p�@�(�@�z�B&  C�                                    By���  �          A+33�У�@@��@�(�B>�RC:��У�@�p�@�{B)�\C޸                                    By�Ƞ  T          A)p���p�@7
=@�p�BA�RC����p�@���@�Q�B-
=CL�                                    By��F  �          A)�����@/\)A z�BF
=Ch�����@�@�z�B1�\CaH                                    By���  �          A*=q��p�@:�H@�ffBA�C���p�@��\@���B,�HC�R                                    By���  
}          A*�H�љ�@N{@��B;C�=�љ�@�33@��HB&�C�                                    By�8  �          A)����@K�@���B=�RC�{����@���@�=qB(  C�)                                    By��  T          A(����G�@33A   BE�RC"�)��G�@^{@�  B5z�C\                                    By� �  
�          A$���Ǯ@
=@��BE=qC!T{�Ǯ@]p�@���B4G�C�q                                    By�/*  
Z          A)���ƸR@��@�\)B3p�C���ƸR@���@��HBC
Y�                                    By�=�  "          A+\)��(�@s�
@�p�B>��C+���(�@�p�@�33B&�C޸                                    By�Lv  
�          A,Q����@x��A (�B@(�CT{���@�Q�@�B'Q�C�                                    By�[  T          A,(����@dz�A=qBD�CxR���@�
=@��
B-  C��                                    By�i�  
�          A+����@_\)A�BE
=C
���@�(�@��
B-��C.                                    By�xh  "          A)���\@VffA33BJ��C
���\@�  @�
=B3\)C��                                    By��  "          A(����
=@Z�H@�p�BD�C33��
=@�Q�@�{B-�Cff                                    By���  
�          A((���33@G�@��RBE�C���33@�
=@���B/�RC�                                    By��Z  
�          A(  ��
=@L��@��\B@�
C� ��
=@���@���B+�C��                                    By��   	�          A(z���G�@Mp�@��B?ffC�q��G�@���@��
B*\)CJ=                                    By���  �          A(����ff@Tz�@�33B@�\C�{��ff@�z�@���B*�C5�                                    By��L  T          A+33���@U@��\B==qC}q���@���@�z�B(�C\                                    By���  �          A,���ə�@�  @��B4�C��ə�@�Q�@��B�C
�{                                    By��  �          A+
=����@��@�z�B �C�����@�{@�G�B

=CǮ                                    By��>  
�          A)p���R@�p�@��B��C
=��R@��@���A��Cٚ                                    By�
�  "          A)G���ff@��@ӅBz�C���ff@�p�@���B��C
=                                    By��  �          A)G�����@j�HA�BN=qC�{����@���@�\)B5�C	�                                    By�(0  �          A)����p�@u@�Q�B-\)C���p�@���@�Q�B�RC�                                    By�6�  "          A(  ���@qG�@�{B,�
Cu����@�{@�ffB�Cs3                                    By�E|  �          A(z���Q�@\��@�Q�B6G�C{��Q�@�p�@�=qB!�RCY�                                    By�T"  �          A.ff��ff@dz�@�33B7�
C����ff@��\@�(�B#G�C+�                                    By�b�  �          A4(��߮@aG�Ap�B7�CG��߮@�=q@�z�B$(�CaH                                    By�qn  �          A6{��p�@33A�\BN�C �3��p�@p��AffB>��C�
                                    By��  
�          A6=q��p�@=p�Az�BBG�C����p�@��H@�p�B0p�C�f                                    By���  �          A7�
�ڏ\@<(�A	G�BDQ�C���ڏ\@�=q@�
=B2p�C�                                    By��`  �          A6�H���@��R@�{B)Q�C����@�p�@�33B\)C�f                                    By��  "          A0z����@P��A\)BA  C�\���@�=q@��B-��C#�                                    By���  T          A(���˅@$z�A z�BF
=C�q�˅@vff@�  B5
=C�=                                    By��R  T          A$(����H@�=q@��B��C@ ���H@�
=@���A�
=C^�                                    By���  �          A
=���?�=q@�p�B?33C%�����@.{@�=qB3�C��                                    By��  
Z          A��ə�?�{@��HB>  C%�f�ə�@/\)@׮B1��Cz�                                    By��D  	`          A (��θR?�  @�{B@{C$�)�θR@;�@��B3p�C��                                    By��  �          A ������?�33@�  BAG�C#� ����@E�@��HB3�HCL�                                    By��  "          A$���Ӆ@33@�B>�HC"���Ӆ@O\)@�B1(�C�H                                    By�!6  
i          A+��ٙ�@33@��B@=qC!L��ٙ�@c33@��B1�RCk�                                    By�/�  E          A*�\���@33A (�BA��C#G����@S33@�\B433C+�                                    By�>�  �          A$���Ϯ@�@�z�B?��C G��Ϯ@`��@�p�B0��C�{                                    By�M(  "          A&�H��
=?��H@�ffBG��C%8R��
=@<(�@�\B;Q�C��                                    By�[�  �          A'33�θR?ǮA ��BI�C&k��θR@3�
@��RB>G�Cz�                                    By�jt  
�          A+��љ�@,��A ��BBp�C�
�љ�@{�@��B2�C�                                    By�y  �          A,����ff@��ABB�RC���ff@l(�@�(�B3�RC+�                                    By���  �          A(���أ�?�33@��BBffC&O\�أ�@7�@��B7{C{                                    By��f  "          A&�R��(�?�\@�33BCQ�C%���(�@>{@�\)B7G�C��                                    By��  "          A)���׮@
=q@�z�B@ffC"G��׮@Vff@�RB2��C��                                    By���  	�          A)G���Q�@Q�@�=qB?Q�C"�=��Q�@S�
@���B2{C��                                    By��X  
�          A����G�?^�R@���BM��C+��G�@�\@�p�BEQ�C!T{                                    By���  "          Az�����?��@�BI�RC)�f����@\)@陚B@z�C�R                                    By�ߤ  "          A (���{?\(�@���BM{C,���{@�\@�=qBE{C!                                    By��J  T          Ap����=u@�BM�C3k����?���@���BI�
C(�f                                    By���  
�          A!����\)?fff@��BF  C,���\)@33@�B>=qC"h�                                    By��  "          A&{���?���@�=qBC�C(L����@#33@��B:{C{                                    By�<  
�          A&�H��\)@\)@�=qB:G�C�3��\)@fff@�B,  C�
                                    By�(�  T          A'\)��p�?�ff@��BC�RC&�f��p�@-p�@�G�B933Cٚ                                    By�7�  "          A&=q��z�?@  @�ffBG�RC-�
��z�?�
=@�  B@�C#��                                    By�F.  
�          A�
��(�?k�@�(�BGG�C+����(�@33@��B?�C"8R                                    By�T�  "          A���G�?�G�@�(�BG�C(����G�@Q�@�B>(�CG�                                    By�cz  T          A!���  @(�@��
B<=qC!\)��  @P��@޸RB/
=C\)                                    By�r   �          A"�R�ҏ\@�R@�p�B;z�C!E�ҏ\@S33@�Q�B.Q�CY�                                    By���  �          A"ff����@p�@�33B9p�C!�H����@P��@�{B,�\C�
                                    By��l  T          A{��{@�\@�(�B8��C u���{@S�
@ָRB+ffC��                                    By��  �          A Q���=q@
=@���B:�C".��=q@I��@�z�B-��C\)                                    By���  T          A"�H��ff@'�@�ffB4(�C����ff@h��@׮B&  C�                                     By��^  �          A"�R��ff@�R@�  B5��C����ff@`  @��B(
=Ck�                                    By��  T          A�\��?���@��BAp�C'@ ��@ ��@��
B7��C��                                    By�ت  �          A�У�?�33@�{B;�RC%Ǯ�У�@*�H@�(�B1Q�C��                                    By��P  T          A�����?У�@�
=B;��C%� ���@'�@�p�B1�Cz�                                    By���  
�          A33�У�@
�H@��HB3ffC!�\�У�@HQ�@θRB'  CY�                                    By��  
�          A���p�@G�@��
B6Q�C#&f��p�@AG�@�Q�B*C�                                    By�B  �          A�����aG�@���BQ�C65�����?Y��@�\)BOC+��                                    By�!�  �          A�R���R���@��
B[Q�CG����R�B�\A ��BcQ�C<B�                                    By�0�  
(          A=q��33�G�@�p�BUQ�C;����33>�33@�ffBV�C0��                                    By�?4  
�          A �����H��A   BRp�C8z����H?&ff@�\)BR  C-�H                                    By�M�  "          A�\��=q?n{@�33B?��C+�f��=q?��H@�z�B8�C#k�                                    By�\�  T          A����Q�?�ff@߮B4�
C'���Q�@ ��@�ffB+�\C�)                                    By�k&  !          A����{>�@���BA��C/���{?��R@�z�B<��C'�                                    By�y�  �          A=q��Q��@�p�BI�RCG���Q쿅�@�(�BR{C>5�                                    By��r  T          A{���9��@��BV�CRp�����ffA�BcCH
=                                    By��  "          Az����H��
@��
B^�HCMz����H����A�Bi��CA��                                    By���  
�          A�\������\@���B^�RCM�\���ÿ���A (�Biz�CB\                                    By��d  
Z          A33���
�\)@�BZG�CO{���
��z�@�ffBe��CD.                                    By��
  �          A�\��p��ff@�
=B^��CM���p����RA�BiG�CB)                                    By�Ѱ  �          A33��33�\)@�G�B^=qCL��33��33A Q�Bh�CAc�                                    By��V  �          Aff����33@���B_\)CJ�f����z�H@��Bh��C?aH                                    By���  T          A����(�� ��@��\B[CO=q��(���
=A��Bg\)CDT{                                    By���  �          A=q��  ���@��
B[�\CM���  ��ffA�BfG�CB��                                    By�H  �          A���\)��@��\B[�
CM
=��\)��  AG�BfG�CB)                                    By��  �          A�R��(���\@��BY�CL���(����HA��Bc�
CAJ=                                    By�)�  �          Az���G���
=@��B[=qCH{��G��Y��A�RBc
=C=�                                    By�8:  �          A���ff���
@��HB_��C?:���ff<��
@��Bb�\C3�=                                    By�F�  �          A������޸R@�BX�RCF�
����8Q�@��RB_C;�                                    By�U�  �          A���������@���B]p�CE  ������@�G�Bc�C9�)                                    By�d,  �          A=q���׿0��@�ffB]z�C;s3����>���@�
=B^�C0ff                                    By�r�  �          A33���\��(�A�HBf
=CJ(����\�^�RA�Bn�C>=q                                    By��x  �          Az���Q��	��AQ�Bf�CLE��Q쿅�A�Bpp�C@\)                                    By��  �          A{����  A��Bd��CL�\������Az�Bn��CA8R                                    By���  �          A�����\�G�@�
=B`CM&f���\���HA33Bj�HCB�                                    By��j  �          A=q��G��
=Ap�Bdp�CK�=��G����A��BmC@E                                    By��  T          A��  ��A ��Bc�CL����  ��\)A(�Bm��CAG�                                    By�ʶ  �          AQ���z��z�A�Bp��CM:���z�uA
�\Bz=qC@Q�                                    By��\  �          A����\)��=qA
�RBv�RCD���\)�L��AQ�B{�C6�{                                    By��  �          A����H��G�A	G�Bm��CEY����H����A33BsQ�C8�R                                    By���  �          A����ÿ��HA�
Bi=qCD+����þ�Q�A	��Bnp�C8)                                    By�N  �          A (����
�˅A��B_�HCDs3���
��A�HBez�C9z�                                    By��  �          A!p���p���(�A�B^�CE����p��&ffA�BdC:��                                    By�"�  �          A!���G���p�@��BK�\CF.��G��z�HA   BRz�C=.                                    By�1@  �          A (����Ϳ�=qAz�B_�CDE���Ϳ�A�\Bd��C9z�                                    By�?�  �          A (���ff��ffAz�Bu\)CJ��ff�.{A�HB}{C<��                                    By�N�  �          A����33��G�A
�RBv33CI����33�&ffA�B}�
C<��                                    By�]2  �          A��  ��z�A��Bq33CJ����  �Q�A�By�C>Q�                                    By�k�  T          A���\)���A	p�BmQ�CI�\��\)�L��A  Bu(�C=��                                    By�z~  �          A!G��������A
�HBm�CHxR�����333AG�Bt��C<G�                                    By��$  �          A"{��z��=qA��Be�CG�\��z�@  A33Bl��C<Q�                                    By���  �          A!���R��{A(�Be�CE:����R���A
ffBkG�C:                                      By��p  �          A z���
=�\A�RBd��CD5���
=���A��Bi�HC9�                                    By��  �          A�R��
=���@��BV�RC@�f��
=���
Ap�BZ�RC7+�                                    By�ü  �          A�������33A\)B]p�C?�q�����\)A��B`�C5k�                                    By��b  �          A (���
=���AQ�B^��CA����
=���RA�Bc(�C7J=                                    By��  �          A������k�@�{BW�C=����=u@��BY=qC3n                                    By��  �          A{��(���\@�\)BM�HC8� ��(�>�(�@�\)BN{C/�                                    By��T  �          A{��\)��ff@�{BV{C@��\)���
A ��BY��C7:�                                    By��  �          A33��\)�Ǯ@�
=BU=qCC:���\)�z�A��BZQ�C9�\                                    By��  T          A\)��33���A (�BVffCF����33�fffA�HB]
=C=!H                                    By�*F  �          A�R��ff��  @�G�BO{CB���ff���@��BS�RC9=q                                    By�8�  �          A����������@�  BP�
CCp������(��@�z�BV{C:z�                                   By�G�  �          AG���p���Q�@��RBNCA����p���\@��\BS33C8�f                                   By�V8  �          A����
=��(�@�ffBM��CA�{��
=�
=q@�=qBR(�C9(�                                   By�d�  �          A
=��
=����@�G�BN�CB����
=�!G�@�p�BSz�C:�                                   By�s�  �          A Q����
��
=@�{BQ��CC�����
�8Q�AG�BW�C;�                                   By��*  �          A (���G���ffA (�BT�CB�R��G��
=A{BY��C9��                                   By���  �          A   ��ff���
A Q�BU\)CEO\��ff�Q�A�RB[\)C<#�                                   By��v  �          A�
����=qA (�BU=qCE�)���^�RA�RB[z�C<��                                   By��  �          A!�Å��{@�{BN��C@�=�Å��
=A ��BR�
C7��                                   By���  �          A!�������(�@��
BL��CAc������@��BP��C8�                                   By��h  �          A!����Q�Ǯ@���BI(�CB  ��Q�#�
@���BM��C9�
                                   By��  �          A ����
=��{@��BI�CB}q��
=�0��@��
BN  C:Y�                                   By��  �          A����
��ff@��BK{CB=q���
�#�
@��BO��C9��                                   By��Z  �          A�\��ff����@�Q�BNQ�CC���ff�0��@���BS\)C:�H                                   By�   �          A33��\)��Q�@���BMz�CC� ��\)�E�@�p�BR�
C;^�                                   By��  �          A �����
���H@���BJ��CC�H���
�L��@�p�BPQ�C;u�                                   By�#L  �          A ���ȣ׿���@�
=BH�CA��ȣ׿��@��\BL�
C9                                     By�1�  �          A�
���H��
=@�G�BL�CA:����H��@���BQ�C8�                                   By�@�  �          A �����H���@��BN{C@�\���H���@�
=BR
=C8p�                                   By�O>  �          A
=�����ffA Q�BW33CCJ=�����RA=qB\(�C::�                                   By�]�  �          A�������A ��BV�HCA�f�����AffB[
=C8��                                    By�l�  �          A!���=q��p�AG�BU�\CBB���=q���A33BZ{C9^�                                    By�{0  �          A!����
�˅A33BZ�CCǮ���
�#�
AG�B_33C:�=                                    By���  T          A!�������\)A  BZ��CD(�����.{A{B`
=C:��                                   By��|  �          A!G���{��\)A�\BX=qCC�H��{�.{A��B]ffC:�
                                   By��"  �          A �����ÿ��HAz�B]p�CB�q���ÿ�\A=qBa��C9E                                   By���  �          A���Q쿾�RA�\B\  CC&f��Q�\)AQ�B`�RC9��                                   By��n  �          A (������A (�BT�
CF^����xQ�A�RB[(�C=��                                   By��  �          A �����׿�33A�B\{CD�����׿8Q�A�Baz�C;c�                                    By��  �          A!��\)���\A��Bg33CA�)��\)����A
{Bk  C7G�                                    By��`  �          A!���(���
=A33BbffCB����(����A��Bf�
C8�q                                   By��  �          A z����Ϳ�33A��B^�CD�����Ϳ5A�HBdffC;�                                    By��  �          A   �������A�
B^�CFǮ����c�
A=qBd\)C=s3                                    By�R  �          A\)���R�ffA ��BW��CI����R��Q�A�B_(�C@T{                                    By�*�  �          A33����!�@��
BH33CKY������@�33BP��CC��                                    By�9�  �          A33��ff��@��
BH�CIaH��ff��(�@��HBO�HCA޸                                    By�HD  �          A!���  �"�\@�z�BF(�CJ���  ��
=@��
BN�\CC�H                                    By�V�  �          A!G���ff�`  @���B4z�CQxR��ff�,(�@�B?��CKz�                                    By�e�  �          A ����(��<(�@���B=��CM����(��
=@�BGp�CF�R                                    By�t6  �          A�R��
=�O\)@�ffB:��CP� ��
=��@�Q�BEp�CJ&f                                    By���  �          A������Dz�@�ffBF=qCP޸�����R@��BQ{CI�
                                    By���  �          A���ff�@��@�z�BCp�CO�
��ff�(�@�p�BM�HCH��                                    By��(  �          A����33�*�H@���BC��CL�\��33����@���BM  CE��                                    By���  �          A����z��!G�@�p�BD�RCK#���z��Q�@���BM(�CD�                                    By��t  �          A����H�7
=@�\)B<=qCM����H�33@�  BE��CF��                                    By��  �          A�����0��@�RB=�RCL�����׿��H@�
=BF�
CF�                                    By���  �          A=q���\��@��RBV\)CE�q���\�p��AB\Q�C=�                                    By��f  �          A\)���R���@�ffBS�CFJ=���R��G�A��BY�C>�                                    By��  T          A!����(��*=q@�\BBp�CK� ��(���=q@�=qBK  CD��                                    By��  �          A"�H��33�z�@�33BA\)CH���33��p�@���BHz�CA�                                    By�X  �          A!���ə���\@�BA��CG��ə����H@�Q�BH�RCA                                    By�#�  T          A"�\��z���
@��B@�CG����z῾�R@�Q�BG(�CA
                                    By�2�  �          A%���{��@�G�B:��CG���{����@�Q�BACAs3                                    By�AJ  �          A$�����
�\)@�B:�CH�����
��@��RBA�CB#�                                    By�O�  �          A#\)���H�(�@��B:�CHT{���H�У�@�(�BAffCA�f                                    By�^�  �          A$����(���@�B<�CF(���(�����@��BCG�C?�=                                    By�m<  �          A#
=��Q���R@�\B@33CE���Q쿕@�  BF  C>.                                    By�{�  �          A �����ÿ��@��BCQ�C=0����þu@�(�BEC6)                                    By���  �          A����p��Y��@��BCz�C;���p��L��@�RBE(�C4s3                                    By��.  �          A!G������@�33B;{CCk�������@�  B@=qC<��                                    By���  
�          A!G��Ϯ��33@�\)B?p�CDL��Ϯ���@�z�BD�C=�=                                    By��z  �          A!��׮��Q�@�RB7�CD��׮��z�@��
B<��C=Ǯ                                    By��   �          A"�\��{���@�\B1�CE+���{����@��B7�C?=q                                    By���  �          A$z����� ��@���B%{CF���녿�\@�Q�B+�CA��                                    By��l  �          A%G����
�	��@�p�B/�RCD�{���
����@�B5�\C>��                                    By��  �          A$����(���@�z�B/z�CDJ=��(����@�=qB5�C>s3                                    By���  �          A'���녿�
=@��HBA�RC?�)��녿
=@�ffBEG�C8�R                                    By ^  �          A(Q��׮����@�ffBD�C?�)�׮�
=qA ��BG�C8��                                    By   �          A(z�����  @��B?{C@8R���+�@�p�BB��C9�                                     By +�  �          A'���  ��
=@��BD=qC=����  ��{@��BF��C6�H                                    By :P  �          A(��������@��B@C=�)����p�@�{BCz�C7
=                                    By H�  �          A)G���p���33@���BA��C=z���p����
@�\)BD33C6��                                    By W�            A&{��33��{@�B?  C?5���33�
=q@���BBQ�C8�                                    By fB  �          A$z����H���@��B8��CCs3���H����@�=qB>�C=&f                                    By t�  �          A$Q��ٙ���=q@�\B>G�C?\�ٙ���@�BA�C8k�                                    By ��  �          A#33�����33@�33B@�\C?�f������@�ffBD�C9#�                                    By �4  �          A$��������@�{BA�RC?�f���\)@���BE(�C8��                                    By ��  �          A#
=���
���@�(�BB(�C?+����
�   @�\)BEffC8Q�                                    By ��  �          A#
=���ÿ��@�B==qC?+����ÿ��@��HB@�\C8�)                                    By �&  �          A#\)�޸R��@��HB7�RC?���޸R�&ff@�{B;=qC9T{                                    By ��  �          A$(��ᙚ���\@��HB6��C>=q�ᙚ�   @�B9��C8
=                                    By �r  �          A$Q���  ��
=@�p�B133C=:���  ��
=@�Q�B3�
C7T{                                    By �  �          A$  ��Q�c�
@�p�B9��C;E��Q��@�
=B;Q�C4�                                    By ��  �          A$  ��Q쿑�@�z�B8��C=@ ��Q쾸Q�@�
=B;33C6�R                                    Byd  �          A#
=��z΅{@�z�B1�C>����z�(�@�B533C8��                                    By
  �          A!���=q��G�@��B1Q�C@
=��=q�B�\@�B5�C:(�                                    By$�  �          A ����33�Ǯ@�G�BL=qCBQ���33�=p�@��BP�C:��                                    By3V  �          A"{���R��z�@��BO��CE�=���R����A=qBUffC>.                                    ByA�  �          A (�����"�\@��BU�\CMY�����ٙ�A\)B^Q�CE�=                                    ByP�  �          A(  ��R��@�Q�B'CE���R���@�RB-�C?��                                    By_H  �          A)����{��G�@�G�B�C@}q��{���@�{B#�C;xR                                    Bym�  �          A)����H�xQ�@�G�B&�RC;����H�k�@�33B(�C5�                                    By|�  �          A(  ��(���(�@�ffB-33C78R��(�>��
@�ffB-\)C1�
                                    By�:  �          A&�\��(��8Q�@�  B!�
C95���(��#�
@�G�B"��C4
                                    By��  �          A%p��33�aG�@ȣ�B�C5� �33>�G�@�Q�B(�C0�3                                    By��  �          A(���z�=�G�@��BQ�C3:��z�?@  @ÅB�C.�3                                    By�,  �          A(���{��ff@�z�B
=C7
�{>W
=@���BffC2��                                    By��  �          A(����
���
@��HBp�C6=q��
>�Q�@��HBffC1z�                                    By�x  �          A(����ͽ�G�@�Q�BffC4� ���?�@�\)BC0
                                    By�  �          A'�
��H=#�
@љ�BC3�q��H?8Q�@�Q�B��C.�q                                    By��  �          A(���=���@�  B��C3L��?L��@�ffB��C.^�                                    By j  �          A'33�����z�@��
B$�\C6&f���>�
=@ۅB$\)C0�                                    By  �          A$  ���@;�@���B
{C�����@`��@�
=B{C�                                    By�  �          A%p�� ��?��@�(�B�RC,� � ��?��H@�\)B�C'��                                    By,\  �          A&{� Q�>��H@�G�B�C0� � Q�?�@�ffB{C+��                                    By;  �          A!���H?z�H@���B�HC,�H���H?У�@�z�B  C(G�                                    ByI�  �          A33��=q�\@ҏ\B133CA
=��=q�Tz�@ָRB5ffC;33                                    ByXN  �          AG���33�(��@�p�B+ffCKc���33��p�@��B3�
CF�                                    Byf�  �          A�����7
=@��
B1��CN5�����(�@�(�B;z�CH��                                    Byu�  �          AG���
=�xQ�@�ffB2Q�C<�
��
=���
@ȣ�B4C6�                                    By�@  �          A
�\��
=����@�Q�B/C7����
=>k�@�Q�B0{C1�f                                    By��  �          A�
��
=�#�
@���B+p�C4\��
=?
=@��B*Q�C.��                                    By��  T          A\)��{=L��@��HB(C3���{?(��@���B'p�C.#�                                    By�2  �          A
=��ff=�@�B&  C2����ff?:�H@�(�B$z�C-�q                                    By��  �          A����{���
@��HB)33C4�f��{?\)@��B(Q�C/:�                                    By�~  �          A�
��Q쿜(�@\B*=qC>�
��Q�z�@�p�B-�C9�                                    By�$  �          A�H�љ���p�@�  B+�
C@���љ��Q�@��
B/��C;�                                    By��  �          A���߮��Q�@�Q�B.p�C=���߮��@�33B1G�C7�                                    By�p  �          A���׿��H@�{B+=qCA����׿�G�@��HB/��C<+�                                    By  �          A���G��A�@��
B!\)CL�
��G���@��B*z�CG�                                    By�  �          A�����;�@�p�B'(�C5�����?�@���B&z�C/��                                    By%b  �          A  ��p���@��RB�
C8ff��p�=�\)@�\)B��C3s3                                    By4  �          A��Q콏\)@�
=B!{C4�)��Q�?�@�ffB G�C/��                                    ByB�  T          A\)��p�?k�@��\Bz�C,s3��p�?�  @�ffB\)C'�\                                    ByQT  �          A\)��p�?Y��@�(�B�RC-���p�?�
=@�Q�B��C(L�                                    By_�  �          A\)��G�@E@\)A�p�CJ=��G�@_\)@h��A�
=C�)                                    Byn�  �          A
=q��p�@q�@dz�A���CW
��p�@�z�@I��A��HC#�                                    By}F  �          A���H��33@�
=Bz�C6�
���H>�  @�\)B�C1��                                    By��  �          A���\��\)@�(�BC6G���\>��R@�(�B�RC1}q                                    By��  �          A����\)>��
@���BffC1}q��\)?c�
@��RB\)C,�                                    By�8  �          Aff��\>�ff@��RB\)C0}q��\?��\@�(�B��C,{                                    By��  �          A����H>�{@�z�B33C1W
���H?h��@��\B(�C,��                                    ByƄ  �          A�����
?Q�@�  B
=C-�H���
?�\)@�(�B	z�C)xR                                    By�*  �          AG�����?��
@���Bz�C(J=����@33@��HB ��C$}q                                    By��  �          A���z�?�
=@�
=B{C'(���z�@��@���A��
C#n                                    By�v  T          A����G�@\)@��
B33C"���G�@/\)@�33A��RCff                                    By  �          A����(�@Q�@���A�  C!���(�@7
=@�(�A�G�C#�                                   By�  �          A
=��@%@�G�A�z�C����@C33@��A�(�C=q                                   Byh  D          @��\�ʏ\@.{@n�RA㙚C� �ʏ\@Fff@Z�HA�G�C��                                   By-  T          A���@{@��
A��C"5���@)��@w
=A�C�                                   By;�  �          A����?˅@�
=B��C'�f��@�@���A�  C#�q                                   ByJZ  
�          A
�\���
?�@���A��C$�����
@��@�G�A�C!n                                   ByY   �          A
�H��{?�G�@�BC*���{?�  @���A��C&O\                                   Byg�  T          A
=��?p��@�{B��C,�)��?�
=@�=qA�  C(��                                   ByvL  �          A����?!G�@���A���C/.���?���@���A�G�C+z�                                   By��  �          A�����33@���BQ�C6�����>8Q�@��B��C2�                                   By��  �          A�H���ͿE�@�{Bp�C9�����;u@�  B�C5�)                                   By�>  
�          A	���{��(�@��B��C>\)��{�&ff@��HB
=C9�
                                   By��  T          Az���{��@���B��C=�)��{�(��@��
B�
C9ff                                   By��  T          A�����aG�@��RA�=qC:� �������@���A�Q�C6�                                   By�0  
�          A��
=����@���A噚C;����
=�(�@��A���C8}q                                   By��  
�          A��R��ff@��A�33C7�)��R�#�
@�z�A���C4{                                   By�|  �          A(��ȣ��@�B�
CFn�ȣ׿�ff@��
B�RCAٚ                                   By�"  �          A�
�ᙚ�   @���BQ�C8{�ᙚ    @�p�B33C3�q                                   By�  	�          A
=����?�
=@|(�A�ffC$\)����@ff@mp�A׮C!8R                                   Byn  �          A�R��Q�@$z�@S33A���C�
��Q�@:=q@@  A���Ch�                                   By&  2          A z�����@%@6ffA��\C� ����@8��@#33A�z�C�                                   By4�  
          A�����@p�@
�HAxQ�C!�)����@+�?��AW�C {                                   ByC`  "          @�\)��G�@�@�
A��\C"�)��G�@!G�@33AmC �                                   ByR  d          @�ff����@(�@n�RA�(�C!������@%�@^{A�G�C��                                   By`�  
�          @���Ϯ@  @s�
A�  C �f�Ϯ@)��@b�\A�Q�C�=                                   ByoR  
          @����(�@��@^�RA�z�Ch���(�@0��@L��AŮC�
                                   By}�  �          @������@6ff@'
=A�\)C�
����@G�@�A��C��                                   By��  "          @�33���H@z�@R�\A���CT{���H@*=q@@��A�33Cu�                                   By�D  "          @�
=���\@.�R@AG�A�=qC����\@B�\@,��A�(�CL�                                   By��  �          @�
=���@!�@A���CE���@1G�@33A�ffCE                                   By��  T          @�\)���H@;�@5�AǙ�CG����H@N{@\)A�Q�C��                                   By�6  �          @ۅ����@�@J�HA���C� ����@-p�@8��Aə�C޸                                   By��  
�          @�{�QG�@%�?fffAB{CǮ�QG�@*=q?��A�\C��                                   By�  "          @����S�
@{�?��
A5G�C!H�S�
@���?
=@�{C}q                                   By�(  �          @����.�R@�=q?��
AX��B��.�R@�?E�A ��B��                                   By�  T          @�Q��,��@�33?�Q�A�p�B�33�,��@�Q�?��A5�B��                                   Byt  T          @�ff�=p�@�
=?��ALz�B�z��=p�@��H?@  @�Q�B�W
                                   By  �          @��R�4z�@��?��A1G�B����4z�@���?��@�G�B��)                                   