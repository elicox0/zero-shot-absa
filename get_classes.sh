grep -o "[[:alpha:]]\+#[[:alpha:]]\+" ../foursquare_gold.xml | sort | uniq | sed 's/\#/ /g' > classes.txt
