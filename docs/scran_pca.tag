<?xml version='1.0' encoding='UTF-8' standalone='yes' ?>
<tagfile doxygen_version="1.12.0">
  <compound kind="file">
    <name>blocked_pca.hpp</name>
    <path>scran_pca/</path>
    <filename>blocked__pca_8hpp.html</filename>
    <class kind="struct">scran_pca::BlockedPcaOptions</class>
    <class kind="struct">scran_pca::BlockedPcaResults</class>
    <namespace>scran_pca</namespace>
  </compound>
  <compound kind="file">
    <name>scran_pca.hpp</name>
    <path>scran_pca/</path>
    <filename>scran__pca_8hpp.html</filename>
    <includes id="simple__pca_8hpp" name="simple_pca.hpp" local="yes" import="no" module="no" objc="no">simple_pca.hpp</includes>
    <includes id="blocked__pca_8hpp" name="blocked_pca.hpp" local="yes" import="no" module="no" objc="no">blocked_pca.hpp</includes>
    <includes id="subset__pca_8hpp" name="subset_pca.hpp" local="yes" import="no" module="no" objc="no">subset_pca.hpp</includes>
    <namespace>scran_pca</namespace>
  </compound>
  <compound kind="file">
    <name>simple_pca.hpp</name>
    <path>scran_pca/</path>
    <filename>simple__pca_8hpp.html</filename>
    <class kind="struct">scran_pca::SimplePcaOptions</class>
    <class kind="struct">scran_pca::SimplePcaResults</class>
    <namespace>scran_pca</namespace>
  </compound>
  <compound kind="file">
    <name>subset_pca.hpp</name>
    <path>scran_pca/</path>
    <filename>subset__pca_8hpp.html</filename>
    <includes id="simple__pca_8hpp" name="simple_pca.hpp" local="yes" import="no" module="no" objc="no">simple_pca.hpp</includes>
    <includes id="blocked__pca_8hpp" name="blocked_pca.hpp" local="yes" import="no" module="no" objc="no">blocked_pca.hpp</includes>
    <namespace>scran_pca</namespace>
  </compound>
  <compound kind="struct">
    <name>scran_pca::BlockedPcaOptions</name>
    <filename>structscran__pca_1_1BlockedPcaOptions.html</filename>
    <templarg>typename EigenVector_</templarg>
    <member kind="variable">
      <type>int</type>
      <name>number</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a05081d55f605a9140315cb29a6c787e2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>scale</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a9a6ae44f80e309c57d337e9f0885a647</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>transpose</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a641144c07ab833ef1129919d736cbbe5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>scran_blocks::WeightPolicy</type>
      <name>block_weight_policy</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a90ea56570afdc1ac9b6d62e26bbab18f</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>scran_blocks::VariableWeightParameters</type>
      <name>variable_block_weight_parameters</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a6ee1dfa151e497faf4c2c436883c5579</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>components_from_residuals</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a5f74122547b03b8cd833d1787211c453</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>realize_matrix</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>ac1ef520bbfe708b6bb8ddf0cab8cccfe</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>af84d32f6a888a991a2dc7fba3139edce</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>irlba::Options&lt; EigenVector_ &gt;</type>
      <name>irlba_options</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a47b4fbf7461a5160001b5f681b87d9f6</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>scran_pca::BlockedPcaResults</name>
    <filename>structscran__pca_1_1BlockedPcaResults.html</filename>
    <templarg>typename EigenMatrix_</templarg>
    <templarg>typename EigenVector_</templarg>
    <member kind="variable">
      <type>EigenMatrix_</type>
      <name>components</name>
      <anchorfile>structscran__pca_1_1BlockedPcaResults.html</anchorfile>
      <anchor>a339a8c10e11d719fdde0b935e56a8bc8</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenVector_</type>
      <name>variance_explained</name>
      <anchorfile>structscran__pca_1_1BlockedPcaResults.html</anchorfile>
      <anchor>ae83cf282975daa64bf304945b6fa71fd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenVector_::Scalar</type>
      <name>total_variance</name>
      <anchorfile>structscran__pca_1_1BlockedPcaResults.html</anchorfile>
      <anchor>a00ac5c254aa7ae41c0a92ada12739a1b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenMatrix_</type>
      <name>rotation</name>
      <anchorfile>structscran__pca_1_1BlockedPcaResults.html</anchorfile>
      <anchor>a442055b2a3841303ec4019394496c057</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenMatrix_</type>
      <name>center</name>
      <anchorfile>structscran__pca_1_1BlockedPcaResults.html</anchorfile>
      <anchor>a97057089c28f6338d93081dabfad2785</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenVector_</type>
      <name>scale</name>
      <anchorfile>structscran__pca_1_1BlockedPcaResults.html</anchorfile>
      <anchor>a910c7e46f4cc53f12b9b15eea7170659</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>irlba::Metrics</type>
      <name>metrics</name>
      <anchorfile>structscran__pca_1_1BlockedPcaResults.html</anchorfile>
      <anchor>a4350c3ad11efb406ff5e2d2ac0c9b041</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>scran_pca::SimplePcaOptions</name>
    <filename>structscran__pca_1_1SimplePcaOptions.html</filename>
    <templarg>typename EigenVector_</templarg>
    <member kind="variable">
      <type>int</type>
      <name>number</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>a5aa1d15f2f308a994b600890ff1d7299</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>scale</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>a0dce270d9989cc7fb4c7e1d34c7612bf</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>transpose</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>ad5081e360db00cab293f8217c7328333</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>realize_matrix</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>a6db1b3773dff9849634fc304b200fff4</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>a97a84e669f11af4363a70e7c17fd1599</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>irlba::Options&lt; EigenVector_ &gt;</type>
      <name>irlba_options</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>af261dda8b883533fb5b2ea4894d8d983</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>scran_pca::SimplePcaResults</name>
    <filename>structscran__pca_1_1SimplePcaResults.html</filename>
    <templarg>typename EigenMatrix_</templarg>
    <templarg>typename EigenVector_</templarg>
    <member kind="variable">
      <type>EigenMatrix_</type>
      <name>components</name>
      <anchorfile>structscran__pca_1_1SimplePcaResults.html</anchorfile>
      <anchor>a1da0c096b49b673dac64a13efca490df</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenVector_</type>
      <name>variance_explained</name>
      <anchorfile>structscran__pca_1_1SimplePcaResults.html</anchorfile>
      <anchor>ad3eda1253739e34d5f5a15778427e3fd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenVector_::Scalar</type>
      <name>total_variance</name>
      <anchorfile>structscran__pca_1_1SimplePcaResults.html</anchorfile>
      <anchor>af715701ecb0d24118de8d704b52e8916</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenMatrix_</type>
      <name>rotation</name>
      <anchorfile>structscran__pca_1_1SimplePcaResults.html</anchorfile>
      <anchor>a3cc8c8bf58ca8fc62eb2bd141f4d6b73</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenVector_</type>
      <name>center</name>
      <anchorfile>structscran__pca_1_1SimplePcaResults.html</anchorfile>
      <anchor>ace27de1e888eaf093e0544ccee7bacd0</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>EigenVector_</type>
      <name>scale</name>
      <anchorfile>structscran__pca_1_1SimplePcaResults.html</anchorfile>
      <anchor>ad9f61059704fb822342fbb776cb66617</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>irlba::Metrics</type>
      <name>metrics</name>
      <anchorfile>structscran__pca_1_1SimplePcaResults.html</anchorfile>
      <anchor>afdc06d82cd7b5b015043fab279159d22</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="namespace">
    <name>scran_pca</name>
    <filename>namespacescran__pca.html</filename>
    <class kind="struct">scran_pca::BlockedPcaOptions</class>
    <class kind="struct">scran_pca::BlockedPcaResults</class>
    <class kind="struct">scran_pca::SimplePcaOptions</class>
    <class kind="struct">scran_pca::SimplePcaResults</class>
    <member kind="typedef">
      <type>SimplePcaOptions&lt; EigenVector_ &gt;</type>
      <name>SubsetPcaOptions</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a30f6e8f0e067aa30d174ba6c4ff05c38</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>SimplePcaResults&lt; EigenMatrix_, EigenVector_ &gt;</type>
      <name>SubsetPcaResults</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a022b0488b926c139064cbff8ae948db2</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>BlockedPcaOptions&lt; EigenVector_ &gt;</type>
      <name>SubsetPcaBlockedOptions</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a5b01976867caaee9e326435554b5007d</anchor>
      <arglist></arglist>
    </member>
    <member kind="typedef">
      <type>BlockedPcaResults&lt; EigenMatrix_, EigenVector_ &gt;</type>
      <name>SubsetPcaBlockedResults</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a07326f9ed36d9caa6328b1c918008a64</anchor>
      <arglist></arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>simple_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a9eaf1d09c8c4fcdf821abd142d7ab8cd</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const SimplePcaOptions&lt; EigenVector_ &gt; &amp;options, SimplePcaResults&lt; EigenMatrix_, EigenVector_ &gt; &amp;output)</arglist>
    </member>
    <member kind="function">
      <type>SimplePcaResults&lt; EigenMatrix_, EigenVector_ &gt;</type>
      <name>simple_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a7d5fdd12c497c154dbce39b3c06cf54e</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const SimplePcaOptions&lt; EigenVector_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>blocked_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a01f33f85f7508d2d1bcc083484ba095e</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const Block_ *block, const BlockedPcaOptions&lt; EigenVector_ &gt; &amp;options, BlockedPcaResults&lt; EigenMatrix_, EigenVector_ &gt; &amp;output)</arglist>
    </member>
    <member kind="function">
      <type>BlockedPcaResults&lt; EigenMatrix_, EigenVector_ &gt;</type>
      <name>blocked_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>aef8d8fcc39408dad09e3aa0048470d56</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const Block_ *block, const BlockedPcaOptions&lt; EigenVector_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>subset_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>ac8c0f28265747d9a8c80dedee1076253</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const SubsetVector_ &amp;subset, const SubsetPcaOptions&lt; EigenVector_ &gt; &amp;options, SubsetPcaResults&lt; EigenMatrix_, EigenVector_ &gt; &amp;output)</arglist>
    </member>
    <member kind="function">
      <type>SubsetPcaResults&lt; EigenMatrix_, EigenVector_ &gt;</type>
      <name>subset_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a9863b52d1424211f123f49b12e99e968</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const SubsetVector_ &amp;subset, const SubsetPcaOptions&lt; EigenVector_ &gt; &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>subset_pca_blocked</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>aff6bb974714400dc60ef955d4da1f2f7</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const SubsetVector_ &amp;subset, const Block_ *block, const SubsetPcaBlockedOptions&lt; EigenVector_ &gt; &amp;options, SubsetPcaBlockedResults&lt; EigenMatrix_, EigenVector_ &gt; &amp;output)</arglist>
    </member>
    <member kind="function">
      <type>SubsetPcaBlockedResults&lt; EigenMatrix_, EigenVector_ &gt;</type>
      <name>subset_pca_blocked</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a16c4e80530a8a1f10692204c5086c440</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const SubsetVector_ &amp;subset, const Block_ *block, const SubsetPcaBlockedOptions&lt; EigenVector_ &gt; &amp;options)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>Principal components analysis, duh</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
