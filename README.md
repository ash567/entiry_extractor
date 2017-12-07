# entiry_extractor

Here I present one use case and the results of an actual experiment.

Suppose I want to mine all the sentenses in a text where one company buys another company. We can provide some templates, which will try to reflect our intenstions. In general Template = {tag1, tag2, tag3 .......}

For our case template would look like: {Company1, buy_verbs, Company2} // There can be more templates, and it is upto the creativity of the user to define templates. But in most cases, the most naive template also works.

We need to tell the model, the meaning of each tag in the template. For that, we provide (concrete real world) seed words to each tag in the temaplate.

In our case:
Company1 = {SanDisk, IBM, Intel, Dell} // Big giant companies
buy_verbs = {bought, buys, buying} // only some words related to buy
Company2 = {Vivisimo, Box} // small startup companies

Here are some of mined senetenses that we got after running the model on AP news corpus.


1. In num US Internet giant [Yahoo] [acquired] [Jordans Maktoob] then the Arab worlds largest online media for an undisclosed fee.
2. Business maker [Deltek] said Thursday that it will [acquire] database and market information [Input Inc] for $num million in an allcash transaction.


Most of the companies in the sentences were not part of the seed which we gave for the tags.
