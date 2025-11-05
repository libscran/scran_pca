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
  <compound kind="struct">
    <name>scran_pca::BlockedPcaOptions</name>
    <filename>structscran__pca_1_1BlockedPcaOptions.html</filename>
    <member kind="variable">
      <type>int</type>
      <name>number</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>aae8f4926b3d8e11588565fa82be60cbd</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>scale</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>abd67f8510e403f541b6275e34f5ad61b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>transpose</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a65498bbdde95073a362a51e942de4fe5</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>scran_blocks::WeightPolicy</type>
      <name>block_weight_policy</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>aa5ac168e8a3384d0fde25a966db0bdac</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>scran_blocks::VariableWeightParameters</type>
      <name>variable_block_weight_parameters</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a300f43e9e22d2f66224f2fbf9611ed4a</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>components_from_residuals</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a3e2503517358ed97e08dc9111e35e791</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>realize_matrix</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a4bb1c0bd64ba1dc70246ce90f941dd1b</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a3ab64262b56d4740ae5ce4501202e857</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>irlba::Options&lt; Eigen::VectorXd &gt;</type>
      <name>irlba_options</name>
      <anchorfile>structscran__pca_1_1BlockedPcaOptions.html</anchorfile>
      <anchor>a5b60e2a25de87678343d8740522f546a</anchor>
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
      <type>bool</type>
      <name>converged</name>
      <anchorfile>structscran__pca_1_1BlockedPcaResults.html</anchorfile>
      <anchor>a0ea80804052ad25021c71c2c506f2f32</anchor>
      <arglist></arglist>
    </member>
  </compound>
  <compound kind="struct">
    <name>scran_pca::SimplePcaOptions</name>
    <filename>structscran__pca_1_1SimplePcaOptions.html</filename>
    <member kind="variable">
      <type>int</type>
      <name>number</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>af99a561e7aa50311bd05ea59802cc492</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>scale</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>abfc39a6f7cd82a793e41d22641dcd0b2</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>transpose</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>a45a65fc15d15e95becb8166f09b0c6fa</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>bool</type>
      <name>realize_matrix</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>a2d20fac36e65846f1431042ac5a48660</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>int</type>
      <name>num_threads</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>a64970ffcc255df6b8bec468be6407a35</anchor>
      <arglist></arglist>
    </member>
    <member kind="variable">
      <type>irlba::Options&lt; Eigen::VectorXd &gt;</type>
      <name>irlba_options</name>
      <anchorfile>structscran__pca_1_1SimplePcaOptions.html</anchorfile>
      <anchor>a3c36f177ab52e01c663d1fea5b90a528</anchor>
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
      <type>bool</type>
      <name>converged</name>
      <anchorfile>structscran__pca_1_1SimplePcaResults.html</anchorfile>
      <anchor>a3dd28009927a0546c3c4e6b5352c3150</anchor>
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
    <member kind="function">
      <type>void</type>
      <name>simple_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>afbe9b43fbd114829551a10897067905c</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const SimplePcaOptions &amp;options, SimplePcaResults&lt; EigenMatrix_, EigenVector_ &gt; &amp;output)</arglist>
    </member>
    <member kind="function">
      <type>SimplePcaResults&lt; EigenMatrix_, EigenVector_ &gt;</type>
      <name>simple_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a91b16a98870f1d46939c2a07c60e6156</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const SimplePcaOptions &amp;options)</arglist>
    </member>
    <member kind="function">
      <type>void</type>
      <name>blocked_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a41ce273ad9e5faaa398afd5dceb88500</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const Block_ *block, const BlockedPcaOptions &amp;options, BlockedPcaResults&lt; EigenMatrix_, EigenVector_ &gt; &amp;output)</arglist>
    </member>
    <member kind="function">
      <type>BlockedPcaResults&lt; EigenMatrix_, EigenVector_ &gt;</type>
      <name>blocked_pca</name>
      <anchorfile>namespacescran__pca.html</anchorfile>
      <anchor>a1d13627678bf5f96d522fcf82400a213</anchor>
      <arglist>(const tatami::Matrix&lt; Value_, Index_ &gt; &amp;mat, const Block_ *block, const BlockedPcaOptions &amp;options)</arglist>
    </member>
  </compound>
  <compound kind="page">
    <name>index</name>
    <title>Principal components analysis, duh</title>
    <filename>index.html</filename>
    <docanchor file="index.html">md__2github_2workspace_2README</docanchor>
  </compound>
</tagfile>
